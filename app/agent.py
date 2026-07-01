# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Generator
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv
# Load local environment variables from .env if present
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.workflow import Workflow
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.genai import types

import sys
from pathlib import Path

# Add project root to sys.path to allow importing from scripts folder
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.dbt_config_models import RootConfig

# Patch all models in scripts.dbt_config_models to use extra="forbid" and override
# model_json_schema to recursively strip "additionalProperties" so that the schema
# is fully compatible with Gemini Developer API mode.
from scripts import dbt_config_models
for attr_name in dir(dbt_config_models):
    attr = getattr(dbt_config_models, attr_name)
    if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
        attr.model_config["extra"] = "forbid"
        attr.model_rebuild(force=True)

original_json_schema = RootConfig.model_json_schema

def custom_model_json_schema(*args, **kwargs):
    schema = original_json_schema(*args, **kwargs)
    def remove_additional_properties(d):
        if isinstance(d, dict):
            d.pop('additionalProperties', None)
            d.pop('additional_properties', None)
            for v in d.values():
                remove_additional_properties(v)
        elif isinstance(d, list):
            for item in d:
                remove_additional_properties(item)
    remove_additional_properties(schema)
    return schema

RootConfig.model_json_schema = custom_model_json_schema

# Schemas for structuring output
class Classification(BaseModel):
    category: Literal["config_only", "model_only", "new_full", "needs_human"] = Field(
        description="The classified category of the Jira ticket."
    )
    reason: str = Field(description="The reasoning behind this classification.")


# Nodes logic
def save_ticket(ctx: Context, node_input: types.Content) -> Event:
    """Extracts raw text from the input content and saves it to workflow state."""
    text = ""
    if node_input and node_input.parts:
        text = "".join(part.text for part in node_input.parts if part.text)
    return Event(
        output=text,
        state={
            "ticket_text": text,
            "validation_attempts": 0,
            "validation_feedback": "",
        }
    )


classifier_agent = LlmAgent(
    name="classifier_agent",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are an expert data engineering assistant. Analyze the incoming Jira ticket text and "
        "classify it into exactly one of the following categories:\n"
        "- config_only: The ticket only requests configuration changes (e.g. updating schedules, tables, metadata in JSON configs, or specifying metadata/parameters for running models without asking to create/modify SQL/Python code files).\n"
        "- model_only: The ticket requests changes only to SQL/Python dbt models.\n"
        "- new_full: The ticket requests a brand new pipeline with both SQL/Python models code AND configurations.\n"
        "- needs_human: The ticket is ambiguous, lacks detail, or doesn't fit the other categories.\n\n"
        "Jira Ticket Text:\n{ticket_text}"
    ),
    output_schema=Classification,
)


def route_ticket(node_input: dict) -> Event:
    """Routes execution based on the classification category."""
    category = node_input.get("category")
    return Event(output=node_input, route=category)


config_generator = LlmAgent(
    name="config_generator",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a dbt DAG config generator. Based on the Jira ticket description, "
        "generate the required dbt DAG configuration conforming to the requested schema.\n\n"
        "Jira Ticket Text:\n{ticket_text}\n"
        "{validation_feedback}"
    ),
    output_schema=RootConfig,
)


def validate_config(ctx: Context, node_input: dict) -> Generator[Event, None, None]:
    """Writes the draft to config.json and runs scripts/validate_dbt_configs.py against it."""
    import subprocess
    attempts = ctx.state.get("validation_attempts", 0) + 1

    filename = "config.json"
    with open(filename, "w") as f:
        json.dump(node_input, f, indent=2)

    script_path = str(Path(__file__).resolve().parent.parent / "scripts" / "validate_dbt_configs.py")

    result = subprocess.run(
        [sys.executable, script_path, filename],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        # Success: print a short summary of what was generated and end.
        dag_configs = node_input.get("dag_configs", [])
        summary = f"Generated {len(dag_configs)} DAG config(s) successfully in `{filename}`."
        for entry in dag_configs:
            dag_cfg = entry.get("dag_config", {})
            job_cfg = entry.get("job_config", {})
            dag_id = dag_cfg.get("dag_id", "unknown_dag")
            steps = job_cfg.get("steps", [])
            summary += f"\n- DAG ID: `{dag_id}` ({len(steps)} step(s))"

        msg = f"Successfully generated and validated dbt DAG config:\n{summary}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=node_input, route="valid")
    else:
        # Failure
        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")

        if attempts >= 4:
            msg = f"Validation failed after 4 attempts. Output:\n```\n{combined_output.strip()}\n```"
            yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
            yield Event(output=combined_output, route="needs_human")
        else:
            validation_feedback = (
                f"\n[Validation Feedback - Attempt {attempts}/4]\n"
                f"The generated config.json failed validation. Please fix the following errors:\n"
                f"```\n{combined_output.strip()}\n```\n"
            )
            yield Event(
                output=node_input,
                route="retry",
                state={
                    "validation_attempts": attempts,
                    "validation_feedback": validation_feedback,
                }
            )


def handle_config_only(node_input: dict) -> Generator[Event, None, None]:
    """Writes the generated config to a config.json file and yields success message."""
    filename = "config.json"
    with open(filename, "w") as f:
        json.dump(node_input, f, indent=2)
    
    msg = f"Successfully generated dbt DAG config in `{filename}`:\n```json\n{json.dumps(node_input, indent=2)}\n```"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output=node_input)


def handle_model_only(node_input: dict) -> Generator[Event, None, None]:
    """Yields 'not implemented yet' message for model_only route."""
    msg = "not implemented yet"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output=msg)


def handle_new_full(node_input: dict) -> Generator[Event, None, None]:
    """Yields 'not implemented yet' message for new_full route."""
    msg = "not implemented yet"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output=msg)


def handle_needs_human(node_input: dict) -> Generator[Event, None, None]:
    """Yields 'not implemented yet' message for needs_human route."""
    msg = "not implemented yet"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output=msg)


# Graph definition
root_agent = Workflow(
    name="dbt_factory_agent",
    edges=[
        ('START', save_ticket),
        (save_ticket, classifier_agent),
        (classifier_agent, route_ticket),
        (route_ticket, {
            'config_only': config_generator,
            'model_only': handle_model_only,
            'new_full': handle_new_full,
            'needs_human': handle_needs_human,
        }),
        (config_generator, validate_config),
        (validate_config, {
            'retry': config_generator,
            'needs_human': handle_needs_human,
        }),
    ]
)

app = App(
    root_agent=root_agent,
    name="app",
)

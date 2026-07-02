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
import sys
from pathlib import Path
from typing import Generator, Any
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv
# Load local environment variables from .env if present
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.workflow import Workflow
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.genai import types
from google.adk.events.request_input import RequestInput

# Load known domains configuration
known_domains_file = Path(__file__).resolve().parent.parent / "scripts" / "known_domains.json"
try:
    with open(known_domains_file, "r") as f:
        KNOWN_DOMAINS = json.load(f)
except Exception:
    KNOWN_DOMAINS = ["sports", "wallet", "analytics", "bi"]

# Load known environments configuration
known_envs_file = Path(__file__).resolve().parent.parent / "scripts" / "known_environments.json"
try:
    with open(known_envs_file, "r") as f:
        KNOWN_ENVS = json.load(f)
except Exception:
    KNOWN_ENVS = ["dev", "stage"]

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


def route_ticket(ctx: Context, node_input: dict) -> Event:
    """Routes execution based on the classification category."""
    category = node_input.get("category")
    route = "needs_human" if category == "needs_human" else "ok"
    return Event(
        output=node_input,
        route=route,
        state={"ticket_category": category}
    )


config_generator = LlmAgent(
    name="config_generator",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a dbt DAG config generator. Based on the Jira ticket description, "
        "generate the required dbt DAG configuration conforming to the requested schema.\n\n"
        "CRITICAL INSTRUCTION: Never invent security/identity values (such as service accounts "
        "like DBT_IMPERSONATE_SERVICE_ACCOUNT or project IDs like DBT_EXECUTION_PROJECT and DBT_PROJECT). "
        "Never invent environment or domain names. "
        "If the ticket does not explicitly provide these values, leave them as empty strings. "
        "Do not guess or hallucinate these fields.\n\n"
        "Jira Ticket Text:\n{ticket_text}\n"
        "{validation_feedback}"
    ),
    output_schema=RootConfig,
)


class DomainSimilarity(BaseModel):
    suggested_domain: str = Field(description="The closest matched known domain from the list of allowed domains, or empty string if it's completely unrelated.")
    reason: str = Field(description="Reasoning for suggestion.")


def check_domain_exact(ctx: Context, node_input: Any) -> Event:
    """Extracts the domain from the ticket text and checks if it's exactly in KNOWN_DOMAINS."""
    from scripts.check_required_fields import parse_env_and_domain
    ticket_text = ctx.state.get("ticket_text", "")
    
    env, domain = parse_env_and_domain(ticket_text)
    
    if not domain:
        return Event(output={"domain": ""}, route="typo_check", state={"domain": ""})
        
    if domain in KNOWN_DOMAINS:
        return Event(
            output=node_input,
            route="ok",
            state={"domain": domain}
        )
    else:
        return Event(
            output={"domain": domain},
            route="typo_check",
            state={"domain": domain}
        )


domain_similarity_agent = LlmAgent(
    name="domain_similarity_agent",
    model="gemini-3.1-flash-lite",
    instruction=(
        f"You are an expert domain validator. You are given a domain name that was parsed from a ticket.\n"
        f"Your task is to compare it against our list of known allowed domains: {', '.join(KNOWN_DOMAINS)}.\n"
        f"If the input domain is a likely typo of one of the known domains (e.g., 'sprot' for 'sports', "
        f"'walet' for 'wallet', 'bi' for 'bi'), return the correct known domain in suggested_domain.\n"
        f"If the domain is completely unrelated and not close to any known domain (e.g. 'banana', 'finance', 'marketing'), "
        f"leave suggested_domain as an empty string.\n\n"
        f"Input Domain:\n{{domain}}"
    ),
    output_schema=DomainSimilarity,
)


def validate_domain_typo_result(ctx: Context, node_input: dict) -> Generator[Event, None, None]:
    """Inspects the similarity agent result and either triggers RequestInput or logs wrong domain."""
    suggested_domain = node_input.get("suggested_domain", "").strip().lower()
    
    if suggested_domain in KNOWN_DOMAINS:
        yield RequestInput(
            interrupt_id=f"domain_typo:{ctx.node_path}",
            message=f"Did you mean '{suggested_domain}'?",
            response_schema=str
        )
        yield Event(state={"suggested_domain": suggested_domain})
    else:
        ticket = ctx.state.get("ticket_text", "")
        entry = {
            "ticket": ticket,
            "parsed_domain": ctx.state.get("domain", ""),
            "reason": f"Unrelated domain suggested: {node_input.get('reason', '')}"
        }
        
        with open("wrong_domain_queue.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        msg = f"Domain validation failed. Logged to wrong_domain_queue.jsonl. Unrelated domain name."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="stop")


def handle_domain_confirmation(ctx: Context, node_input: Any) -> Event:
    """Handles the user's confirmation response for the domain typo."""
    response = str(node_input).strip().lower()
    
    if response in ("yes", "y", "confirm", "true"):
        suggested = ctx.state.get("suggested_domain", "")
        return Event(
            output=suggested,
            route="ok",
            state={"domain": suggested}
        )
    else:
        msg = "User declined domain correction. Stopping execution."
        return Event(
            output=msg,
            route="stop",
            content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
        )


class EnvSimilarity(BaseModel):
    suggested_env: str = Field(description="The closest matched known environment from the list of allowed environments, or empty string if it's completely unrelated.")
    reason: str = Field(description="Reasoning for suggestion.")


def check_env_exact(ctx: Context, node_input: Any) -> Event:
    """Extracts environment, runs production guard, checks exact match or routes to typo check."""
    from scripts.check_required_fields import parse_env_and_domain
    ticket_text = ctx.state.get("ticket_text", "")
    
    env, domain = parse_env_and_domain(ticket_text)
    
    if env in ("prod", "production", "live"):
        msg = "this repo holds non-production config only, so production must be handled by a human."
        return Event(
            output=msg,
            route="prod_guard",
            state={"missing_critical_fields": ["environment (production guard)"]}
        )
        
    if not env:
        return Event(output={"env": ""}, route="typo_check", state={"env": ""})
        
    if env in KNOWN_ENVS:
        return Event(
            output=node_input,
            route="ok",
            state={"environment": env}
        )
    else:
        return Event(
            output={"env": env},
            route="typo_check",
            state={"env": env}
        )


env_similarity_agent = LlmAgent(
    name="env_similarity_agent",
    model="gemini-3.1-flash-lite",
    instruction=(
        f"You are an expert environment validator. You are given an environment name that was parsed from a ticket.\n"
        f"Your task is to compare it against our list of known allowed environments: {', '.join(KNOWN_ENVS)}.\n"
        f"If the input environment is a likely typo of one of the known environments (e.g., 'stag' or 'staging' for 'stage', "
        f"'develop' or 'deve' for 'dev'), return the correct known environment in suggested_env.\n"
        f"If the environment is completely unrelated and not close to any known environment (e.g. 'xyz', 'production', 'live'), "
        f"leave suggested_env as an empty string.\n\n"
        f"Input Environment:\n{{env}}"
    ),
    output_schema=EnvSimilarity,
)


def validate_env_typo_result(ctx: Context, node_input: dict) -> Generator[Event, None, None]:
    """Inspects the similarity agent result and either triggers RequestInput or logs wrong environment."""
    suggested_env = node_input.get("suggested_env", "").strip().lower()
    
    if suggested_env in KNOWN_ENVS:
        yield RequestInput(
            interrupt_id=f"env_typo:{ctx.node_path}",
            message=f"Did you mean '{suggested_env}'?",
            response_schema=str
        )
        yield Event(state={"suggested_env": suggested_env})
    else:
        ticket = ctx.state.get("ticket_text", "")
        entry = {
            "ticket": ticket,
            "parsed_env": ctx.state.get("env", ""),
            "reason": f"Unrelated environment suggested: {node_input.get('reason', '')}"
        }
        
        with open("wrong_domain_queue.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        msg = f"Environment validation failed. Logged to wrong_domain_queue.jsonl. Unrelated environment name."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="stop")


def handle_env_confirmation(ctx: Context, node_input: Any) -> Event:
    """Handles the user's confirmation response for the environment typo."""
    response = str(node_input).strip().lower()
    
    if response in ("yes", "y", "confirm", "true"):
        suggested = ctx.state.get("suggested_env", "")
        return Event(
            output=suggested,
            route="ok",
            state={"environment": suggested}
        )
    else:
        msg = "User declined environment correction. Stopping execution."
        return Event(
            output=msg,
            route="stop",
            content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
        )


def dispatch_by_category(ctx: Context, node_input: Any) -> Event:
    """Dispatches execution to the category handler after domain validation succeeds."""
    category = ctx.state.get("ticket_category")
    return Event(output=node_input, route=category)


def check_critical_fields(ctx: Context, node_input: dict) -> Event:
    """Checks for critical identity/security fields and environment/domain settings."""
    from scripts.check_required_fields import check_config_and_env
    ticket_text = ctx.state.get("ticket_text", "")
    resolved_domain = ctx.state.get("domain")
    resolved_env = ctx.state.get("environment")
    
    res = check_config_and_env(node_input, ticket_text, resolved_domain, resolved_env)
    missing = res["missing_fields"]
    is_prod = res["is_prod"]
    
    if missing:
        msg = f"Critical fields/metadata are missing: {', '.join(missing)}"
        return Event(
            output=msg,
            route="needs_human",
            state={"missing_critical_fields": missing},
            content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
        )
    elif is_prod:
        msg = "this repo holds non-production config only, so production must be handled by a human."
        return Event(
            output=msg,
            route="needs_human",
            state={"missing_critical_fields": ["environment (production guard)"]},
            content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
        )
    else:
        resolved_path = res["resolved_path"]
        exists_str = "exists" if res["path_exists"] else "does not exist"
        log_msg = f"Resolved target path: {resolved_path} ({exists_str})"
        return Event(
            output=node_input,
            route="ok",
            content=types.Content(role='model', parts=[types.Part.from_text(text=log_msg)])
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


# model_builder LlmAgent: generates ONLY text (dbt SQL & _schema.yml).
# It must never execute code, touch files, or run git directly.
model_builder = LlmAgent(
    name="model_builder",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a dbt model code builder. Your job is to generate ONLY two files in your output text block:\n"
        "1. The dbt SQL model code (SELECT queries only).\n"
        "2. The schema metadata file (_schema.yml).\n\n"
        "Rules:\n"
        "- Output the two files strictly using standard markdown code blocks with the format:\n"
        "```sql\n"
        "-- filepath: dbt/models/public/<model_name>.sql\n"
        "select ...\n"
        "```\n"
        "and\n"
        "```yaml\n"
        "# filepath: dbt/models/public/_schema.yml\n"
        "version: 2\n"
        "...\n"
        "```\n"
        "- You must only generate SELECT queries. Never generate any write operations like DROP, DELETE, TRUNCATE, ALTER, GRANT, INSERT, UPDATE, or CREATE OR REPLACE.\n"
        "- Do not execute any code, shell commands, or make network calls."
    )
)

def prepare_model_builder_input(ctx: Context, node_input: Any) -> Event:
    """Formats the ticket description and domain for model_builder."""
    ticket_text = ctx.state.get("ticket_text", "")
    domain = ctx.state.get("domain", "")
    prompt = (
        f"Domain: {domain}\n"
        f"Ticket Description: {ticket_text}\n\n"
        f"Please generate the dbt model SQL and the _schema.yml file contents according to the rules."
    )
    return Event(output=prompt)

def validate_and_push_model(ctx: Context, node_input: str) -> Generator[Event, None, None]:
    """
    Parses LLM output, performs a deterministic SQL safety check,
    and pushes the files to a feature branch in a secure temp directory.
    """
    import re
    import subprocess
    import tempfile
    import os
    import time
    from scripts.check_sql_safety import check_sql_safety
    from dotenv import load_dotenv

    text = node_input or ""

    # Parse SQL and YAML blocks from markdown
    sql_match = re.search(r'```sql\s*([\s\S]*?)```', text)
    yaml_match = re.search(r'```yaml\s*([\s\S]*?)```', text)

    sql_content = sql_match.group(1).strip() if sql_match else ""
    yaml_content = yaml_match.group(1).strip() if yaml_match else ""

    if not sql_content:
        # Invalid generation or missing SQL block
        msg = "Model generation failed: No SQL block found in LLM response."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="needs_human", state={"missing_critical_fields": ["SQL block generation"]})
        return

    # Extract model name
    model_name = "new_model"
    name_match = re.search(r'filepath:\s*(?:dbt/)?models/public/([a-zA-Z0-9_]+)\.sql', text)
    if name_match:
        model_name = name_match.group(1)
    else:
        name_match_in_sql = re.search(r'filepath:\s*(?:dbt/)?models/public/([a-zA-Z0-9_]+)\.sql', sql_content)
        if name_match_in_sql:
            model_name = name_match_in_sql.group(1)

    # Guard Layer 2: Deterministic SQL Safety Check
    # This check ensures the generated code only contains SELECT queries.
    # A SELECT-only query cannot modify tables, delete data, or change permissions.
    is_safe, safety_reason = check_sql_safety(sql_content)
    if not is_safe:
        msg = f"SQL safety check rejected the model code:\nReason: {safety_reason}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="needs_human", state={"missing_critical_fields": ["SQL safety check"]})
        return

    # Guard Layer 3: Git Workspace Isolation (tempfile.TemporaryDirectory)
    # The git operations run inside a clean temp directory with a localized clone.
    # If anything fails or raises an exception, the directory is automatically removed
    # from the local disk, cleaning up any generated state/dirty files.
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")

    # Secure remote URL: if token is present in env, use token for authentication.
    # Otherwise, fall back to SSH for local testing.
    # Never log or print the token or remote credentials to avoid exposing secrets.
    if github_token:
        repo_url = f"https://x-access-token:{github_token}@github.com/hasan-tavakoli/dv-sports-etl.git"
    else:
        repo_url = "git@github.com:hasan-tavakoli/dv-sports-etl.git"

    # Guard Layer 3.1: Feature branch only
    # Pushing to feature branch ensures it must go through a pull request and review,
    # and cannot directly alter the master/staging branches.
    feature_branch = f"feature/add-model-{model_name}-{int(time.time())}"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repo
            clone_cmd = ["git", "clone", repo_url, "."]
            res = subprocess.run(clone_cmd, cwd=temp_dir, capture_output=True, text=True)
            if res.returncode != 0:
                err = res.stderr.replace(github_token, "***") if github_token else res.stderr
                raise RuntimeError(f"Git clone failed: {err}")

            # Create feature branch
            subprocess.run(["git", "checkout", "-b", feature_branch], cwd=temp_dir, check=True)

            # Write files
            # SQL model file
            sql_file_path = os.path.join(temp_dir, "dbt", "models", "public", f"{model_name}.sql")
            os.makedirs(os.path.dirname(sql_file_path), exist_ok=True)
            with open(sql_file_path, "w") as f:
                f.write(sql_content)

            # YAML schema file
            yaml_file_path = os.path.join(temp_dir, "dbt", "models", "public", "_schema.yml")
            if yaml_content:
                with open(yaml_file_path, "w") as f:
                    f.write(yaml_content)

            # Commit changes
            subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
            subprocess.run(["git", "commit", "-m", f"✨ feat: generate dbt model {model_name}"], cwd=temp_dir, check=True)

            # Push changes to feature branch
            push_cmd = ["git", "push", "origin", feature_branch]
            push_res = subprocess.run(push_cmd, cwd=temp_dir, capture_output=True, text=True)
            if push_res.returncode != 0:
                err = push_res.stderr.replace(github_token, "***") if github_token else push_res.stderr
                raise RuntimeError(f"Git push failed: {err}")

        # Outside the temp directory, it has been successfully cleaned up
        msg = f"Successfully generated, safety checked, and pushed dbt model `{model_name}` to feature branch `{feature_branch}`."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="ok")

    except Exception as e:
        msg = f"Model deployment failed during git push: {str(e)}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="needs_human", state={"missing_critical_fields": ["Git Push Exception"]})



def handle_new_full(node_input: dict) -> Generator[Event, None, None]:
    """Yields 'not implemented yet' message for new_full route."""
    msg = "not implemented yet"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output=msg)


def handle_needs_human(ctx: Context, node_input: Any) -> Generator[Event, None, None]:
    """Appends the ticket and missing fields to a local queue and yields message."""
    ticket = ctx.state.get("ticket_text", "")
    missing_fields = ctx.state.get("missing_critical_fields", [])
    
    entry = {
        "ticket": ticket,
        "missing_critical_fields": missing_fields,
        "reason": str(node_input)
    }
    
    with open("needs_human_queue.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
        
    msg = f"Ticket routed to human review. Logged to needs_human_queue.jsonl. Reason: {node_input}"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output=entry)


# Graph definition
root_agent = Workflow(
    name="dbt_factory_agent",
    edges=[
        ('START', save_ticket),
        (save_ticket, classifier_agent),
        (classifier_agent, route_ticket),
        (route_ticket, {
            'ok': check_domain_exact,
            'needs_human': handle_needs_human,
        }),
        (check_domain_exact, {
            'ok': check_env_exact,
            'typo_check': domain_similarity_agent,
        }),
        (domain_similarity_agent, validate_domain_typo_result),
        (validate_domain_typo_result, handle_domain_confirmation),
        (handle_domain_confirmation, {
            'ok': check_env_exact,
            'stop': handle_needs_human,
        }),
        (check_env_exact, {
            'ok': dispatch_by_category,
            'typo_check': env_similarity_agent,
            'prod_guard': handle_needs_human,
        }),
        (env_similarity_agent, validate_env_typo_result),
        (validate_env_typo_result, handle_env_confirmation),
        (handle_env_confirmation, {
            'ok': dispatch_by_category,
            'stop': handle_needs_human,
        }),
        (dispatch_by_category, {
            'config_only': config_generator,
            'model_only': prepare_model_builder_input,
            'new_full': handle_new_full,
        }),
        (prepare_model_builder_input, model_builder),
        (model_builder, validate_and_push_model),
        (validate_and_push_model, {
            'needs_human': handle_needs_human,
        }),
        (config_generator, check_critical_fields),
        (check_critical_fields, {
            'ok': validate_config,
            'needs_human': handle_needs_human,
        }),
        (validate_config, {
            'retry': config_generator,
            'needs_human': handle_needs_human,
        }),
    ]
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)

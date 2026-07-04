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

def reject_ticket(reason_category: str, reason_text: str, ticket: str, user_id: str) -> None:
    """Publishes the rejection to Pub/Sub and conditionally comments on Jira if user_id starts with jira-."""
    import json
    import os
    import logging
    import httpx
    from google.cloud import pubsub_v1

    logger = logging.getLogger("dbt_factory_agent")
    logger.info(f"reject_ticket called: category={reason_category}, user_id={user_id}, reason={reason_text}")

    # 1. Publish to Pub/Sub topic "dv-rejected-tickets" in project ht-project-500813
    project_id = "ht-project-500813"
    topic_name = "dv-rejected-tickets"
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)
        payload = {
            "category": reason_category,
            "reason": reason_text,
            "ticket": ticket,
            "user_id": user_id
        }
        data_str = json.dumps(payload)
        data_bytes = data_str.encode("utf-8")
        future = publisher.publish(topic_path, data=data_bytes)
        msg_id = future.result()
        logger.info(f"Published rejection payload to {topic_name}. Message ID: {msg_id}")
    except Exception as e:
        logger.error(f"Failed to publish rejection to Pub/Sub: {e}")

    # 2. Comment back to Jira if user_id starts with "jira-"
    if user_id and user_id.startswith("jira-"):
        issue_key = user_id.split("jira-", 1)[1]
        jira_base_url = os.getenv("JIRA_BASE_URL", "https://hassan-t.atlassian.net").rstrip("/")
        jira_email = os.getenv("JIRA_EMAIL")
        jira_api_token = os.getenv("JIRA_API_TOKEN")

        if not jira_email or not jira_api_token:
            logger.warning("Jira credentials (JIRA_EMAIL or JIRA_API_TOKEN) are missing. Skipping Jira comment.")
            return

        comment_url = f"{jira_base_url}/rest/api/3/issue/{issue_key}/comment"
        
        # Build ADF body
        message_text = f"Ticket rejected.\nCategory: {reason_category}\nReason: {reason_text}"
        body = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": message_text
                            }
                        ]
                    }
                ]
            }
        }

        try:
            logger.info(f"Posting comment to Jira issue {issue_key}...")
            response = httpx.post(
                comment_url,
                json=body,
                auth=(jira_email, jira_api_token),
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 201:
                logger.info(f"Successfully posted rejection comment to Jira issue {issue_key}.")
            else:
                logger.error(f"Failed to post comment to Jira. Status: {response.status_code}, Body: {response.text}")
        except Exception as e:
            logger.error(f"Error while posting comment to Jira issue {issue_key}: {e}")


# Schemas for structuring output
class Classification(BaseModel):
    category: Literal["config_only", "model_only", "new_full", "needs_human"] = Field(
        description="The classified category of the Jira ticket."
    )
    reason: str = Field(description="The reasoning behind this classification.")


class VibeDiffSummary(BaseModel):
    plain_summary: str = Field(
        description="2-3 plain-English sentences explaining what this change does in the project."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="One of low, medium, or high."
    )
    risk_reason: str = Field(
        description="A one-line reason explaining the chosen risk level."
    )
    intent_alignment: str = Field(
        description="A short sentence explaining how well the generated model matches the ticket's intent."
    )


class IntentPayload(BaseModel):
    service_account: str | None = Field(
        default=None,
        description="The GCP service account to impersonate (e.g., service-account@project.iam.gserviceaccount.com). Leave empty if not found."
    )
    execution_project: str | None = Field(
        default=None,
        description="The GCP project where the job executes (e.g. execution-project-id). Leave empty if not found."
    )
    target_project: str | None = Field(
        default=None,
        description="The GCP target project/DBT_PROJECT (e.g. target-project-id). Leave empty if not found."
    )
    dag_id: str | None = Field(
        default=None,
        description="The Airflow DAG ID (e.g. daily_active_customers_dag). Leave empty if not found."
    )
    schedule: str | None = Field(
        default=None,
        description="The cron schedule (e.g. '0 6 * * *'). Leave empty if not found."
    )
    config_intent: str | None = Field(
        default=None,
        description="Any specific config intent/wishes explicitly requested by the user, such as omitting a step (e.g., 'no public step') or custom behavior. Leave empty if there is no explicit structural exception."
    )


# Nodes logic
def save_ticket(ctx: Context, node_input: types.Content) -> Event:
    """Extracts raw text from the input content and saves it to workflow state.
    
    If a pending_schedule_payload exists in session state, the user is responding
    to a schedule prompt — short-circuit to the schedule handler instead of
    restarting the full pipeline.
    """
    text = ""
    if node_input and node_input.parts:
        text = "".join(part.text for part in node_input.parts if part.text)
    
    # Check if we are resuming from a schedule prompt
    pending = ctx.state.get("pending_schedule_payload")
    if pending:
        return Event(
            output=text,
            route="resume_schedule",
        )
    
    return Event(
        output=text,
        route="normal",
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
        "- model_only: The ticket requests changes to SQL/Python dbt models. Even if it contains metadata parameters like service account, projects, schedule, or config intents to accompany the model, as long as it does NOT explicitly ask to create, edit, or update separate configuration JSON/YAML files (like config.json or deploy.yml), it belongs to model_only.\n"
        "  IMPORTANT: Providing schedule, service account, execution project, target project, or similar metadata ALONGSIDE a model request is NORMAL and EXPECTED for model_only. That metadata alone — no matter how much of it is present — must NEVER push the classification to new_full. It only describes how the model's pipeline should run; it is not a request to touch a separate config file.\n"
        "- new_full: The ticket requests a brand new pipeline with BOTH SQL/Python model code AND an EXPLICIT, separate request to create, edit, or update a config file. The ticket must name or clearly describe a distinct config artifact (e.g. \"also create config.json\", \"add a deploy.yml entry\", \"set up the DAG's configuration file\") as something to be created/edited IN ADDITION TO the model code. Simply listing schedule/service account/project values is NOT sufficient — see model_only above.\n"
        "- needs_human: The ticket is ambiguous, lacks detail, or doesn't fit the other categories.\n\n"
        "If you are genuinely unsure whether a ticket is model_only or new_full, choose model_only.\n\n"
        "Examples:\n"
        "(a) \"Add a new dbt model for the sports domain that computes daily active users. "
        "Schedule: 0 6 * * *. Service account: analytics-dev@... Execution project: dv-dev-eu-w1-sports-elt. "
        "Target project: dv-dev-eu-w1-sports-data.\" -> model_only. The metadata accompanies the model "
        "request but nothing asks to create or edit a config file.\n"
        "(b) \"Add a new dbt model for the sports domain AND create its config.json and deploy.yml.\" "
        "-> new_full. There is an explicit, separate request to create config files in addition to the model.\n\n"
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
        reason_text = f"Unrelated domain suggested: {node_input.get('reason', '')}"
        
        reject_ticket(
            reason_category="invalid_domain",
            reason_text=reason_text,
            ticket=ticket,
            user_id=ctx.user_id
        )

        entry = {
            "ticket": ticket,
            "parsed_domain": ctx.state.get("domain", ""),
            "reason": reason_text
        }
        
        with open("wrong_domain_queue.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        msg = f"Domain validation failed. Logged to wrong_domain_queue.jsonl. Unrelated domain name."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="stop")


def handle_domain_confirmation(ctx: Context, node_input: Any) -> Event:
    """Handles the user's confirmation response for the domain typo.
    
    Proceeds with the suggested domain on affirmatives, overrides with a valid
    known domain if the user typed it explicitly, and stops on negatives or unrecognized input.
    """
    response = str(node_input).strip().lower()
    
    if response in ("yes", "y", "confirm", "true"):
        suggested = ctx.state.get("suggested_domain", "")
        return Event(
            output=suggested,
            route="ok",
            state={"domain": suggested}
        )
    elif response in KNOWN_DOMAINS:
        return Event(
            output=response,
            route="ok",
            state={"domain": response}
        )
    elif response in ("no", "n", "cancel", "false"):
        msg = "User declined domain correction. Stopping execution."
        return Event(
            output=msg,
            route="stop",
            content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
        )
    else:
        msg = f"Unrecognized domain response '{response}'. Stopping execution."
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
        reason_text = f"Unrelated environment suggested: {node_input.get('reason', '')}"
        
        reject_ticket(
            reason_category="invalid_environment",
            reason_text=reason_text,
            ticket=ticket,
            user_id=ctx.user_id
        )

        entry = {
            "ticket": ticket,
            "parsed_env": ctx.state.get("env", ""),
            "reason": reason_text
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


def publish_config_only_payload(ctx: Context, node_input: Any) -> Generator[Event, None, None]:
    """
    Publishes the config_only payload to Pub/Sub topic 'dv-model-image-ready'
    in project 'ht-project-500813'.
    
    The payload is wrapped in the streamQuery envelope required by the config-agent:
    {
      "class_method": "stream_query",
      "input": {
        "message": "<the config_only payload JSON as a string>",
        "user_id": "orchestrator"
      }
    }
    """
    import json
    from google.cloud import pubsub_v1
    from google.adk.events.event import Event
    from google.genai import types

    payload_str = str(node_input)
    
    # Wrap in the stream_query envelope matching what the CI/CD pipeline does
    envelope = {
        "class_method": "stream_query",
        "input": {
            "message": payload_str,
            "user_id": "orchestrator"
        }
    }
    
    project_id = "ht-project-500813"
    topic_name = "dv-model-image-ready"
    
    msg = f"Publishing config_only event to Pub/Sub topic '{topic_name}' in project '{project_id}'..."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)
        
        envelope_str = json.dumps(envelope)
        data_bytes = envelope_str.encode("utf-8")
        
        # Publish synchronously (wait for the future)
        future = publisher.publish(topic_path, data=data_bytes)
        message_id = future.result()
        
        success_msg = (
            f"Config change queued (Pub/Sub Message ID: {message_id}) — "
            "the config-agent will open a PR shortly."
        )
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=success_msg)]))
        yield Event(output=success_msg)
        
    except Exception as e:
        err_msg = f"Failed to publish config_only event to Pub/Sub: {e}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=err_msg)]))
        yield Event(output={"error": err_msg})





intent_extractor = LlmAgent(
    name="intent_extractor",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are an expert data engineering assistant. Analyze the incoming Jira ticket text and "
        "extract the configuration fields requested. If any field is not mentioned in the ticket, "
        "leave it as empty or null. Do not invent any values.\n\n"
        "Jira Ticket Text:\n{ticket_text}"
    ),
    output_schema=IntentPayload,
)

model_intent_extractor = LlmAgent(
    name="model_intent_extractor",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are an expert data engineering assistant. Analyze the incoming Jira ticket text and "
        "extract the configuration fields requested. If any field is not mentioned in the ticket, "
        "leave it as empty or null. Do not invent any values.\n\n"
        "Jira Ticket Text:\n{ticket_text}"
    ),
    output_schema=IntentPayload,
)


def validate_config_only_payload(ctx: Context, node_input: IntentPayload) -> Generator[Event, None, None]:
    """
    Validates completeness of the extracted payload for config_only / model_only category.
    
    Key logic:
    1. Derive dag_id from domain if empty (pattern: dv_<domain>_elt).
    2. Check if the DAG already exists in the config repo (standalone path check).
       - EXISTS → schedule comes from existing config, don't require it from the user.
       - NEW → if schedule is missing, ask the user (stop and prompt, don't guess).
    3. Never invent schedule, service accounts, or projects — read from config or ask.
    """
    from scripts.check_required_fields import check_config, resolve_target_path
    
    ticket_text = ctx.state.get("ticket_text", "")
    resolved_domain = ctx.state.get("domain")
    resolved_env = ctx.state.get("environment")
    category = ctx.state.get("ticket_category")
    
    # Get values from IntentPayload
    service_account = getattr(node_input, "service_account", "") or ""
    execution_project = getattr(node_input, "execution_project", "") or ""
    target_project = getattr(node_input, "target_project", "") or ""
    dag_id = getattr(node_input, "dag_id", "") or ""
    schedule = getattr(node_input, "schedule", "") or ""
    config_intent = getattr(node_input, "config_intent", "") or ""
    
    # ── Step 1: Derive dag_id if empty ──────────────────────────────────────
    if not dag_id.strip() and resolved_domain:
        dag_id = f"dv_{resolved_domain}_elt"
        
    # ── Step 2: Standalone path-exists check ────────────────────────────────
    # This runs independently of the missing-fields check so we always know
    # whether we're adding to an existing DAG or creating a new one.
    dag_exists = False
    if resolved_env and resolved_domain and dag_id.strip():
        is_prod = resolved_env in ("prod", "production")
        if not is_prod:
            try:
                _, dag_exists = resolve_target_path(
                    resolved_env, resolved_domain, dag_id, ticket_text
                )
            except ValueError:
                pass  # unsupported environment — will be caught later
    
    # ── Step 3: Check identity/security fields ──────────────────────────────
    missing = []
    if not service_account.strip():
        missing.append("service_account")
    if not execution_project.strip():
        missing.append("execution_project")
    if not target_project.strip():
        missing.append("target_project")
        
    # Category-specific field requirements
    if category == "model_only":
        # model_only requires at least one of: dag_id OR domain
        has_domain = resolved_domain and resolved_domain.strip()
        has_dag_id = dag_id and dag_id.strip()
        if not has_domain and not has_dag_id:
            missing.append("domain or dag_id")
    else:
        # config_only requires dag_id explicitly
        if not dag_id.strip():
            missing.append("dag_id")
            
    # ── Step 4: Schedule requirement ────────────────────────────────────────
    # If DAG exists → schedule comes from existing config, don't require it.
    # If DAG is new → schedule is required.
    if not dag_exists:
        if not schedule.strip():
            if missing:
                # Other critical fields also missing → needs_human with all issues
                missing.append("schedule")
            else:
                # Only schedule is missing on a NEW DAG → ask the user.
                # Store the pending payload in session state so save_ticket can
                # detect it on re-entry and route directly to handle_schedule_response.
                pending_payload = {
                    "domain": resolved_domain,
                    "environment": resolved_env,
                    "dag_id": dag_id,
                    "service_account": service_account,
                    "execution_project": execution_project,
                    "target_project": target_project,
                    "config_intent": config_intent,
                    "category": category
                }
                msg = (
                    f"This is a new DAG ('{dag_id}') and no schedule was given "
                    "— what cron schedule should it use? (e.g. '0 6 * * *')"
                )
                yield Event(
                    content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]),
                )
                yield Event(
                    output=msg,
                    route="ask_schedule",
                    state={"pending_schedule_payload": pending_payload}
                )
                return
    
    # ── Step 5: Prod guard ──────────────────────────────────────────────────
    is_prod = resolved_env in ("prod", "production")
    
    if missing:
        msg = f"Critical fields/metadata are missing: {', '.join(missing)}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(
            output=msg,
            route="needs_human",
            state={"missing_critical_fields": missing}
        )
        return
        
    if is_prod:
        msg = "this repo holds non-production config only, so production must be handled by a human."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(
            output=msg,
            route="needs_human",
            state={"missing_critical_fields": ["environment (production guard)"]}
        )
        return
        
    # ── Step 6: Assemble payload ────────────────────────────────────────────
    payload = {
        "source": "model" if category == "model_only" else "config_only",
        "domain": resolved_domain,
        "environment": resolved_env,
        "dag_id": dag_id,
        "schedule": schedule,
        "service_account": service_account,
        "execution_project": execution_project,
        "target_project": target_project,
        "config_intent": config_intent
    }
    
    import json
    if category == "model_only":
        log_msg = f"Extracted metadata for model PR:\n```json\n{json.dumps(payload, indent=2)}\n```"
        print(log_msg)
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=log_msg)]))
        yield Event(
            output=node_input,
            route="ok_model",
            state={"agent_metadata": payload}
        )
    else:
        payload_str = json.dumps(payload, indent=2)
        log_msg = f"Assembled payload for config-agent:\n```json\n{payload_str}\n```"
        print(log_msg)
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=log_msg)]))
        yield Event(output=payload_str, route="ok")


def stop_for_user_input(node_input: Any) -> Event:
    """Terminal node — halts the workflow so the user can respond.
    
    The workflow stops here. When the user sends their next message,
    save_ticket detects pending_schedule_payload in session state and
    routes directly to handle_schedule_response.
    """
    return Event(output=node_input)


def handle_schedule_response(ctx: Context, node_input: Any) -> Event:
    """Processes the user's schedule response and resumes the workflow.
    
    Called when save_ticket detects pending_schedule_payload in session state.
    The user's response (a cron expression) arrives as the node_input string.
    After assembling the payload, clear pending_schedule_payload so future
    messages go through the normal pipeline.
    """
    response_schedule = str(node_input).strip()
    
    pending = ctx.state.get("pending_schedule_payload", {})
    category = pending.get("category")
    
    # Assemble payload with the user-provided schedule
    payload = {
        "source": "model" if category == "model_only" else "config_only",
        "domain": pending.get("domain"),
        "environment": pending.get("environment"),
        "dag_id": pending.get("dag_id"),
        "schedule": response_schedule,
        "service_account": pending.get("service_account"),
        "execution_project": pending.get("execution_project"),
        "target_project": pending.get("target_project"),
        "config_intent": pending.get("config_intent")
    }
    
    import json
    if category == "model_only":
        log_msg = f"Extracted metadata for model PR:\n```json\n{json.dumps(payload, indent=2)}\n```"
        print(log_msg)
        return Event(
            output=node_input,
            route="ok_model",
            # Clear pending_schedule_payload so future messages go through normal pipeline
            state={"agent_metadata": payload, "pending_schedule_payload": None},
            content=types.Content(role='model', parts=[types.Part.from_text(text=log_msg)])
        )
    else:
        payload_str = json.dumps(payload, indent=2)
        log_msg = f"Assembled payload for config-agent:\n```json\n{payload_str}\n```"
        print(log_msg)
        return Event(
            output=payload_str,
            route="ok",
            state={"pending_schedule_payload": None},
            content=types.Content(role='model', parts=[types.Part.from_text(text=log_msg)])
        )


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
    """Formats the ticket description and domain for model_builder.

    Also stashes the IntentPayload metadata from model_intent_extractor into
    state, since it would otherwise be lost while model_builder/
    check_sql_safety_early operate on plain SQL text — validate_config_only_payload
    needs it back once the SQL safety check passes.
    """
    ticket_text = ctx.state.get("ticket_text", "")
    domain = ctx.state.get("domain", "")
    prompt = (
        f"Domain: {domain}\n"
        f"Ticket Description: {ticket_text}\n\n"
        f"Please generate the dbt model SQL and the _schema.yml file contents according to the rules."
    )
    if hasattr(node_input, "model_dump"):
        intent_payload = node_input.model_dump()
    elif isinstance(node_input, dict):
        intent_payload = node_input
    else:
        intent_payload = {}
    return Event(output=prompt, state={"model_intent_payload": intent_payload})

def check_sql_safety_early(ctx: Context, node_input: str) -> Generator[Event, None, None]:
    """Parses model_builder output, runs early SQL safety check, and routes accordingly."""
    import re
    from scripts.check_sql_safety import check_sql_safety

    text = node_input or ""

    # Parse SQL block from markdown
    sql_match = re.search(r'```sql\s*([\s\S]*?)```', text)
    sql_content = sql_match.group(1).strip() if sql_match else ""

    if not sql_content:
        msg = "Model generation failed: No SQL block found in LLM response."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="unsafe", state={"missing_critical_fields": ["SQL block generation"]})
        return

    # Deterministic SQL Safety Check
    is_safe, safety_reason = check_sql_safety(sql_content)
    if not is_safe:
        ticket = ctx.state.get("ticket_text", "")
        reject_ticket(
            reason_category="unsafe_sql",
            reason_text=safety_reason,
            ticket=ticket,
            user_id=ctx.user_id
        )
        
        msg = f"SQL safety check rejected the model code:\nReason: {safety_reason}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="unsafe", state={"missing_critical_fields": ["SQL safety check"]})
        return

    # If safe, store in state and proceed
    yield Event(
        output=ctx.state.get("model_intent_payload", {}),
        route="safe",
        state={"generated_text": text}
    )


def validate_and_push_model(ctx: Context, node_input: Any) -> Generator[Event, None, None]:
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

    text = ctx.state.get("generated_text", "") or ""

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

            # Write .agent-metadata.json at REPO ROOT if present
            agent_metadata = ctx.state.get("agent_metadata")
            if agent_metadata:
                metadata_file_path = os.path.join(temp_dir, ".agent-metadata.json")
                with open(metadata_file_path, "w") as f:
                    json.dump(agent_metadata, f, indent=2)

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
        yield Event(
            output=msg,
            route="ok",
            state={
                "generated_sql": sql_content,
                "generated_yaml": yaml_content,
                "model_name": model_name,
                "feature_branch": feature_branch
            }
        )

    except Exception as e:
        msg = f"Model deployment failed during git push: {str(e)}"
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg, route="needs_human", state={"missing_critical_fields": ["Git Push Exception"]})


def prepare_pr_summarizer_input(ctx: Context, node_input: Any) -> Event:
    """Prepares prompt for the PR Vibe Diff summarizer."""
    ticket_text = ctx.state.get("ticket_text", "")
    sql = ctx.state.get("generated_sql", "")
    yaml = ctx.state.get("generated_yaml", "")
    
    prompt = (
        f"Original Ticket:\n{ticket_text}\n\n"
        f"Generated dbt SQL Model:\n```sql\n{sql}\n```\n\n"
        f"Generated schema.yml:\n```yaml\n{yaml}\n```\n\n"
        f"Please analyze these inputs and generate the structured PR review summary."
    )
    return Event(output=prompt)


# pr_summarizer LlmAgent: generates the structured vibe diff summary
pr_summarizer = LlmAgent(
    name="pr_summarizer",
    model="gemini-3.1-flash-lite",
    output_schema=VibeDiffSummary,
    instruction=(
        "You are an expert code reviewer. Your job is to analyze the generated dbt model and schema "
        "relative to the original ticket intent, and produce a structured Vibe Diff summary."
    )
)


def create_pull_request(ctx: Context, node_input: VibeDiffSummary) -> Generator[Event, None, None]:
    """
    Deterministic step: parses the VibeDiffSummary from the LLM, builds the markdown PR body,
    and calls the GitHub API to create a Pull Request.
    """
    import json
    import urllib.request
    import urllib.error
    
    plain_summary = getattr(node_input, "plain_summary", "")
    risk_level = getattr(node_input, "risk_level", "low")
    risk_reason = getattr(node_input, "risk_reason", "")
    intent_alignment = getattr(node_input, "intent_alignment", "")
    
    model_name = ctx.state.get("model_name", "new_model")
    feature_branch = ctx.state.get("feature_branch", "")
    sql = ctx.state.get("generated_sql", "")
    yaml = ctx.state.get("generated_yaml", "")
    
    # Build PR Body in Markdown format
    pr_body = (
        f"## Summary\n"
        f"{plain_summary}\n\n"
        f"## Risk\n"
        f"- **Level**: {risk_level.upper()}\n"
        f"- **Reason**: {risk_reason}\n\n"
        f"## Intent alignment\n"
        f"{intent_alignment}\n\n"
        f"## Security check\n"
        f"- **Status**: SELECT-only confirmed\n"
        f"- **Detail**: The SQL code contains only SELECT statements and has been deterministically verified to contain no DDL/DML write statements.\n\n"
        f"## Files changed\n"
        f"### `dbt/models/public/{model_name}.sql`\n"
        f"```sql\n{sql}\n```\n\n"
        f"### `dbt/models/public/_schema.yml`\n"
        f"```yaml\n{yaml}\n```"
    )
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        msg = (
            f"GITHUB_TOKEN not found in env. Simulating Pull Request creation.\n\n"
            f"=== PR Title ===\n"
            f"✨ feat: add dbt model {model_name}\n\n"
            f"=== PR Base/Head ===\n"
            f"Base: staging\n"
            f"Head: {feature_branch}\n\n"
            f"=== PR Body ===\n"
            f"{pr_body}"
        )
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg)
        return

    # Call GitHub API to create PR
    url = "https://api.github.com/repos/hasan-tavakoli/dv-sports-etl/pulls"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "dbt-factory-agent",
        "Content-Type": "application/json"
    }
    payload = {
        "title": f"✨ feat: add dbt model {model_name}",
        "head": feature_branch,
        "base": "staging",
        "body": pr_body
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            pr_url = res_data.get("html_url", "")
            
            msg = (
                f"Successfully created Pull Request:\n"
                f"PR URL: {pr_url}\n\n"
                f"### Vibe Diff Pull Request Body\n"
                f"---\n"
                f"{pr_body}"
            )
            yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
            yield Event(output={"pr_url": pr_url, "pr_body": pr_body})
            
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        msg = f"Failed to create GitHub Pull Request. HTTP Error: {e.code}. Details: {err_msg}"
        if github_token:
            msg = msg.replace(github_token, "***")
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
        yield Event(output=msg)



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
        (save_ticket, {
            'normal': classifier_agent,
            'resume_schedule': handle_schedule_response,
        }),
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
        (validate_domain_typo_result, {
            'stop': handle_needs_human,
            '__DEFAULT__': handle_domain_confirmation,
        }),
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
        (validate_env_typo_result, {
            'stop': handle_needs_human,
            '__DEFAULT__': handle_env_confirmation,
        }),
        (handle_env_confirmation, {
            'ok': dispatch_by_category,
            'stop': handle_needs_human,
        }),
        (dispatch_by_category, {
            'config_only': intent_extractor,
            'model_only': model_intent_extractor,
            'new_full': handle_new_full,
        }),
        (intent_extractor, validate_config_only_payload),
        (model_intent_extractor, prepare_model_builder_input),
        (prepare_model_builder_input, model_builder),
        (model_builder, check_sql_safety_early),
        (check_sql_safety_early, {
            'unsafe': handle_needs_human,
            'safe': validate_config_only_payload,
        }),
        (validate_config_only_payload, {
            'ok_model': validate_and_push_model,
            'ok': publish_config_only_payload,
            'ask_schedule': stop_for_user_input,
            'needs_human': handle_needs_human,
        }),
        # handle_schedule_response is reached via save_ticket -> resume_schedule
        (handle_schedule_response, {
            'ok_model': validate_and_push_model,
            'ok': publish_config_only_payload,
        }),
        (validate_and_push_model, {
            'ok': prepare_pr_summarizer_input,
            'needs_human': handle_needs_human,
        }),
        (prepare_pr_summarizer_input, pr_summarizer),
        (pr_summarizer, create_pull_request),
    ]
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)

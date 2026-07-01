"""
Pydantic schema — the SINGLE SOURCE OF TRUTH for a valid dbt DAG config.json.

Why this file exists (Shift Intelligence Left):
    We do NOT ask the LLM to "remember" what a valid config looks like. Instead
    the LLM generates a draft, and this deterministic schema decides pass/fail.
    Deterministic rules belong in code, not in prompts — this saves tokens and
    removes a whole class of hallucination failures.

Shape decision:
    The real dv-platform-config repo wraps everything in a LIST under the key
    "dag_configs". A single file may therefore declare more than one DAG entry.
    Each entry has exactly two sections: dag_config (orchestration) and
    job_config (execution). We mirror that shape exactly so the validator
    accepts files that look like the real repo, and rejects anything else.

Design note on `extra`:
    env_variables and dbt_flags are intentionally open-ended (extra="allow")
    because teams add domain-specific keys (DBT_GCS_DL_BUCKET, custom flags,
    etc.). The REQUIRED keys below are the ones the DAG factory cannot run
    without — those are enforced strictly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Orchestration(BaseModel):
    """
    Cross-DAG dependency block (the three-signal / dataset-event system).
    Optional: a producer DAG with no upstreams simply omits depends_on.
    """

    model_config = ConfigDict(extra="allow")

    # List of upstream DAG IDs this DAG waits for. Empty list = no dependency.
    depends_on: list[str] | None = None
    # Cron fallback if no upstream dataset event ever arrives.
    fallback_schedule: str | None = None
    # Parse-time on/off switch for signal emission.
    enabled: bool | None = None


class DagConfig(BaseModel):
    """
    Orchestration settings for one DAG: identity, schedule, tags, docs.
    These map 1:1 to the dag_config section in the real repo.
    """

    model_config = ConfigDict(extra="allow")

    # Base DAG id; the factory appends a region suffix (e.g. _eu_w1) at build time.
    dag_id: str
    # Cron expression, e.g. "30 0 * * *".
    schedule: str
    # ISO date string, e.g. "2025-10-01".
    start_date: str
    # Timezone for the schedule. Defaults to UTC in the factory if omitted.
    tz_info: str | None = None
    # Airflow backfill behaviour.
    catchup: bool | None = None
    # Airflow UI tags for filtering.
    tags: list[str] | None = None
    # Markdown shown in the Airflow UI.
    doc_md: str | None = None
    # Optional cross-DAG dependency block.
    orchestration: Orchestration | None = None


class DbtFlags(BaseModel):
    """
    dbt CLI flags for one step, e.g. {"--select": "path:models/public"}.
    Open-ended because teams use --select, --exclude, --full-refresh, etc.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Step(BaseModel):
    """
    One dbt execution unit inside a DAG. Steps run in listed order.
    """

    model_config = ConfigDict(extra="allow")

    # Human-readable id, surfaced in Airflow task ids.
    step_name: str
    # Prefixes for dynamic task mapping; empty/absent = single task.
    task_dataset_prefix: list[str] | None = None
    # dbt command to run; the factory defaults to "build" if omitted.
    dbt_invocation_command: str | None = None
    # Per-step CLI flags.
    dbt_flags: DbtFlags | None = None
    # Override default source variables.
    source_vars: list[str] | None = None


class EnvVariables(BaseModel):
    """
    Environment variables passed to the Cloud Run container.

    Only the keys the factory genuinely cannot run without are REQUIRED. The
    reason each is required is given inline so the model (and any human reading
    a validation error) understands WHY, not just THAT, the rule exists.
    """

    model_config = ConfigDict(extra="allow")  # domain-specific keys welcome

    # Required: which BigQuery project pays for / runs the queries.
    DBT_EXECUTION_PROJECT: str
    # Required: cross-project access identity. Without it the job cannot auth.
    DBT_IMPERSONATE_SERVICE_ACCOUNT: str
    # Required: the target data warehouse project the models land in.
    DBT_PROJECT: str
    # Required: which profile from profiles.yml to use (e.g. "cloud-run").
    DBT_PROFILE: str
    # Required: BigQuery region, e.g. "europe-west1".
    DBT_LOCATION: str
    # Strongly conventional but not fatal if missing — kept optional.
    DBT_SCHEMA: str | None = None
    DBT_DOMAIN_NAME: str | None = None


class JobConfig(BaseModel):
    """
    Execution settings for one DAG: container env + ordered steps.
    """

    model_config = ConfigDict(extra="allow")

    env_variables: EnvVariables
    # A DAG with zero steps does nothing, so at least one is required.
    steps: list[Step] = Field(min_length=1)
    run_time_variables: dict[str, str] | None = None


class DagEntry(BaseModel):
    """
    One complete DAG definition = orchestration + execution.
    """

    dag_config: DagConfig
    job_config: JobConfig


class RootConfig(BaseModel):
    """
    Top-level shape of a config.json file.

    Matches the real dv-platform-config repo: a LIST of DAG entries under the
    key "dag_configs". At least one entry is required — an empty file is never
    valid.
    """

    dag_configs: list[DagEntry] = Field(min_length=1)

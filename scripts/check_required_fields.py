#!/usr/bin/env python3
"""
Deterministic validator for critical identity, security fields, and environment/domain logic.
Checks both config content and ticket text.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_env_and_domain(ticket_text: str) -> tuple[str | None, str | None]:
    """
    Parses environment and domain from the ticket text.
    """
    # 1. Match environment: "Environment: stage" or "env: dev" or keyword matching
    env_match = re.search(r'(?i)\benv(?:ironment)?\s*[:=]\s*(\w+)', ticket_text)
    env = env_match.group(1).lower() if env_match else None
    
    if not env:
        if re.search(r'(?i)\bproduction\b|\bprod\b', ticket_text):
            env = "prod"
        elif re.search(r'(?i)\bstage\b', ticket_text):
            env = "stage"
        elif re.search(r'(?i)\bdev\b', ticket_text):
            env = "dev"
            
    # 2. Match domain: "Domain: sports" or "domain = sports"
    domain_match = re.search(r'(?i)\bdomain\s*[:=]\s*(\w+)', ticket_text)
    domain = domain_match.group(1).lower() if domain_match else None
    
    return env, domain


def resolve_target_path(env: str, domain: str, dag_id: str, ticket_text: str) -> tuple[str, bool]:
    """
    Resolves target path: <subtree>/<domain>/<dag-name>/config.json
    where dag-name uses hyphens instead of underscores.
    """
    if env == "dev":
        subtree = "leo-dev-eu"
    elif env == "stage":
        if re.search(r'(?i)\bsouth-america\b|\bsa\b', ticket_text):
            subtree = "leo-stage-sa"
        else:
            subtree = "leo-stage-eu"
    else:
        raise ValueError(f"Unsupported environment: {env}")
        
    dag_name_hyphenated = dag_id.replace("_", "-")
    target_path = Path(subtree) / domain / dag_name_hyphenated / "config.json"
    
    return str(target_path), target_path.exists()


def check_config(data: dict) -> list[str]:
    """
    Backward-compatibility wrapper for check_config.
    """
    missing = []
    dag_configs = data.get("dag_configs", [])
    if not dag_configs:
        return ["DBT_IMPERSONATE_SERVICE_ACCOUNT", "DBT_EXECUTION_PROJECT", "DBT_PROJECT"]

    for entry in dag_configs:
        job_config = entry.get("job_config", {})
        env_vars = job_config.get("env_variables", {}) if job_config else {}
        for field in ["DBT_IMPERSONATE_SERVICE_ACCOUNT", "DBT_EXECUTION_PROJECT", "DBT_PROJECT"]:
            val = env_vars.get(field)
            if not val or not str(val).strip():
                if field not in missing:
                    missing.append(field)
    return missing


def check_config_and_env(data: dict, ticket_text: str, resolved_domain: str | None = None, resolved_env: str | None = None) -> dict:
    """
    Checks for both critical fields in config and environment/domain in ticket text.
    """
    missing = []
    
    parsed_env, parsed_domain = parse_env_and_domain(ticket_text)
    env = resolved_env if resolved_env else parsed_env
    domain = resolved_domain if resolved_domain else parsed_domain
    
    if not env:
        missing.append("environment")
    if not domain:
        missing.append("domain")
        
    # Check other critical fields in config
    dag_configs = data.get("dag_configs", [])
    if not dag_configs:
        missing.extend(["DBT_IMPERSONATE_SERVICE_ACCOUNT", "DBT_EXECUTION_PROJECT", "DBT_PROJECT"])
    else:
        for entry in dag_configs:
            job_config = entry.get("job_config", {})
            env_vars = job_config.get("env_variables", {}) if job_config else {}
            for field in ["DBT_IMPERSONATE_SERVICE_ACCOUNT", "DBT_EXECUTION_PROJECT", "DBT_PROJECT"]:
                val = env_vars.get(field)
                if not val or not str(val).strip():
                    if field not in missing:
                        missing.append(field)
                        
    is_prod = env in ("prod", "production")
    
    resolved_path = None
    path_exists = False
    if not missing and not is_prod:
        dag_cfg = dag_configs[0].get("dag_config", {})
        dag_id = dag_cfg.get("dag_id", "unknown_dag")
        resolved_path, path_exists = resolve_target_path(env, domain, dag_id, ticket_text)
        
    return {
        "missing_fields": missing,
        "is_prod": is_prod,
        "resolved_path": resolved_path,
        "path_exists": path_exists,
        "environment": env,
        "domain": domain
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("check_required_fields.py: no config file provided", file=sys.stderr)
        return 1

    path = Path(argv[0])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading config: {exc}", file=sys.stderr)
        return 1

    # For CLI execution we check config and env (if passed as second arg, else empty)
    ticket_text = argv[1] if len(argv) > 1 else ""
    res = check_config_and_env(data, ticket_text)
    
    if res["missing_fields"]:
        print(f"Missing critical fields: {', '.join(res['missing_fields'])}")
        return 1
    if res["is_prod"]:
        print("Error: Production configuration must be handled by a human.")
        return 1
        
    if res["resolved_path"]:
        exists_str = "exists" if res["path_exists"] else "does not exist"
        print(f"Resolved path: {res['resolved_path']} ({exists_str})")
        
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

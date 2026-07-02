#!/usr/bin/env python3
"""
Deterministic validator for dbt DAG config.json files.

Usage (called by the ADK validate node, by pre-commit, or by hand):
    python scripts/validate_dbt_configs.py path/to/config.json [more.json ...]

Exit codes:
    0  — every file is valid
    1  — at least one file failed

Why this is a plain script and not an LLM node (Shift Intelligence Left):
    JSON parsing and schema checking are deterministic. Asking a model to do
    them wastes tokens and can hallucinate a "pass". A 60-line script gives a
    hard, repeatable answer.

Self-heal contract:
    When a file is invalid, we print each error as a single line prefixed with
    "  - " and formatted as "[field -> path] message". This exact, structured
    output is fed straight back to the config-generator node so it can correct
    the specific field that failed, instead of regenerating blindly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The schema import is relative to this scripts/ folder. The ADK node adds
# scripts/ to sys.path before calling, and pre-commit runs from repo root, so
# we support both by trying a direct import first, then a package-style one.
try:
    from dbt_config_models import RootConfig
except ImportError:  # pragma: no cover - fallback when run from repo root
    sys.path.insert(0, str(Path(__file__).parent))
    from dbt_config_models import RootConfig

from pydantic import ValidationError


def describe_path(path: Path) -> str:
    """Turn a config path into a short label like 'my-dag in dv-dev-eu'."""
    parts = path.parts
    if len(parts) >= 3:
        return f"{parts[-2]} in {parts[-3]}"
    return str(path)


def validate_file(path: Path) -> list[str]:
    """Return a list of human-readable error strings, or [] if the file is valid."""
    errors: list[str] = []

    # Step 1: can we even read it?
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"  - [file] cannot read file: {exc}"]

    # Step 2: is it valid JSON at all?
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"  - [json] invalid JSON: {exc}"]

    # Step 3: does it match the schema (the source of truth)?
    try:
        RootConfig.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            errors.append(f"  - [{loc}] {err['msg']}")

    return errors


def main(argv: list[str]) -> int:
    if not argv:
        print("validate_dbt_configs.py: no files to check", file=sys.stderr)
        return 0

    any_failed = False
    for arg in argv:
        path = Path(arg)
        label = describe_path(path)
        errors = validate_file(path)
        if errors:
            any_failed = True
            print(f"FAIL  {label}")
            for line in errors:
                print(line)
        else:
            print(f"OK    {label}")

    if any_failed:
        print("\nValidation failed. Fix the fields listed above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

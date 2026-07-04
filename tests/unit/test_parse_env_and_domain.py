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
"""
Covers scripts.check_required_fields.parse_env_and_domain against common Jira
markup wrappers ({{monospace}}, *bold*, _italic_, `code`, quotes) around the
"Domain:"/"Environment:" label and/or value, which used to leave the captured
value stuck with its wrapping markup (or empty, since \\w+ can't match the
"{" that starts "{{sprot}}").
"""

import pytest

from scripts.check_required_fields import parse_env_and_domain


@pytest.mark.parametrize(
    "ticket_text,expected_domain",
    [
        ("Domain: sprot", "sprot"),
        ("Domain: {{sprot}}", "sprot"),
        ("Domain: *sprot*", "sprot"),
        ("Domain: `sprot`", "sprot"),
        ('Domain: "sprot"', "sprot"),
        ("domain = sprot", "sprot"),
        ("domain=sprot", "sprot"),
        ("Domain:sprot", "sprot"),
        ("* Domain: {{sprot}}", "sprot"),
        ("Domain: dv-sports-elt", "dv-sports-elt"),
    ],
)
def test_domain_extraction(ticket_text: str, expected_domain: str) -> None:
    _, domain = parse_env_and_domain(ticket_text)
    assert domain == expected_domain


def test_environment_with_monospace_markup() -> None:
    env, _ = parse_env_and_domain("Environment: {{stage}}")
    assert env == "stage"


def test_domain_and_environment_together_with_markup() -> None:
    ticket_text = "* Domain: {{sprot}}\n* Environment: {{stage}}"
    env, domain = parse_env_and_domain(ticket_text)
    assert domain == "sprot"
    assert env == "stage"


def test_bold_wrapped_label_including_separator() -> None:
    # The label AND the colon are both inside the bold markers.
    env, domain = parse_env_and_domain("*Domain:* sprot")
    assert domain == "sprot"


def test_braces_wrapped_label() -> None:
    env, domain = parse_env_and_domain("{{Domain}}: sprot")
    assert domain == "sprot"


def test_trailing_prose_after_value_is_not_captured() -> None:
    _, domain = parse_env_and_domain("Domain: sprot (typo for sports?)")
    assert domain == "sprot"


def test_no_domain_label_present_returns_none() -> None:
    env, domain = parse_env_and_domain("This ticket is about the sprot domain.")
    assert domain is None
    # Bare "prod"/"stage"/"dev" keyword scan still applies to environment.
    assert env is None


def test_environment_prose_fallback_still_works_without_a_label() -> None:
    env, _ = parse_env_and_domain("Please deploy this to stage as soon as possible.")
    assert env == "stage"


def test_production_guard_keyword_still_detected() -> None:
    env, _ = parse_env_and_domain("This must go straight to production.")
    assert env == "prod"


def test_full_realistic_ticket_with_markup() -> None:
    ticket_text = """Summary: Update the schedule for the sprot domain ELT

Context: This ticket concerns the sprot domain in the stage environment.

Acceptance Criteria:
- DAG ID dv_sports_elt
- Service account: analytics-dev@dv-dev-eu-w1-sports-elt.iam.gserviceaccount.com
- Execution project: dv-dev-eu-w1-sports-elt
- Target project: dv-dev-eu-w1-sports-data

Other information:
* Domain: {{sprot}}
* Environment: {{stage}}
"""
    env, domain = parse_env_and_domain(ticket_text)
    assert domain == "sprot"
    assert env == "stage"


def test_real_production_ticket_with_no_newlines_and_multiple_braces_fields() -> None:
    # Exact text reported from production: Jira's h3./bullet wiki-markup
    # flattened into a single line (no \n at all) with several other
    # {{...}}-wrapped fields (DAG ID, schedule, service account, projects)
    # appearing before the Domain/Environment fields at the very end.
    ticket_text = (
        "Update schedule for the sprot pipeline h3. Summary  Update the "
        "schedule for the sprot domain ELT so it runs earlier.  h3. Context  "
        "The change is for the sprot domain in the stage environment.  "
        "h3. Acceptance criteria  * Update the schedule for DAG ID "
        "{{dv_sports_elt}}. * Change the schedule to {{0 4 * * *}}. "
        "* Use the service account "
        "{{analytics-dev@dv-dev-eu-w1-sports-elt.iam.gserviceaccount.com}}. "
        "* Keep the execution project as {{dv-dev-eu-w1-sports-elt}}. "
        "* Keep the target project as {{dv-dev-eu-w1-sports-data}}.  "
        "h3. Other information  * Domain: {{sprot}} * Environment: {{stage}}"
    )
    assert "\n" not in ticket_text

    env, domain = parse_env_and_domain(ticket_text)
    assert domain == "sprot"
    assert env == "stage"

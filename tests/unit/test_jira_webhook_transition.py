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
Covers jira_webhook.main's transition check (should_process_event) and
duplicate-delivery de-dup (evaluate_webhook_event). The webhook used to fire
on any event where the issue's CURRENT status was "In Progress" (a state
check), which caused two production bugs: reprocessing an already-"In
Progress" issue touched by an unrelated event, and an infinite loop via the
orchestrator's own rejection comment (which triggers an "issue updated"
event with no status change). Both are fixed by requiring THIS event's own
changelog to show a transition into "In Progress".
"""

import logging

import pytest

from jira_webhook.main import (
    _recent_transition_keys,
    evaluate_webhook_event,
    should_process_event,
)


@pytest.fixture(autouse=True)
def _clear_dedup_cache():
    """Isolate tests from each other's dedup state - it's shared module state."""
    _recent_transition_keys.clear()
    yield
    _recent_transition_keys.clear()


def _status_changelog_item(from_string: str, to_string: str) -> dict:
    return {
        "field": "status",
        "fieldtype": "jira",
        "from": "1",
        "fromString": from_string,
        "to": "3",
        "toString": to_string,
    }


def _updated_payload(
    issue_key: str,
    current_status: str,
    changelog_items: list,
    changelog_id: str = "10001",
    webhook_event: str = "jira:issue_updated",
) -> dict:
    return {
        "webhookEvent": webhook_event,
        "timestamp": 1720000000000,
        "issue": {
            "key": issue_key,
            "fields": {
                "summary": "Test ticket",
                "description": "Test description",
                "status": {"name": current_status},
            },
        },
        "changelog": {"id": changelog_id, "items": changelog_items},
    }


def test_status_transition_to_in_progress_is_processed():
    payload = _updated_payload(
        issue_key="SCRUM-10",
        current_status="In Progress",
        changelog_items=[_status_changelog_item("To Do", "In Progress")],
        changelog_id="ch-1",
    )
    should_process, reason = should_process_event(payload)
    assert should_process is True
    assert "SCRUM-10" in reason


def test_already_in_progress_refire_is_ignored():
    # The changelog's own status item shows no real transition (from == to).
    payload = _updated_payload(
        issue_key="SCRUM-11",
        current_status="In Progress",
        changelog_items=[_status_changelog_item("In Progress", "In Progress")],
        changelog_id="ch-2",
    )
    should_process, _ = should_process_event(payload)
    assert should_process is False


def test_comment_only_event_is_ignored():
    # This is the loop guard: the orchestrator's own rejection comment looks
    # exactly like this - a comment event on an issue that's "In Progress",
    # with no changelog at all.
    payload = {
        "webhookEvent": "comment_created",
        "timestamp": 1720000000000,
        "issue": {
            "key": "SCRUM-12",
            "fields": {
                "summary": "Test ticket",
                "description": "Test description",
                "status": {"name": "In Progress"},
            },
        },
        "comment": {"body": "Ticket rejected.\nCategory: unsafe_sql\nReason: ..."},
    }
    should_process, reason = should_process_event(payload)
    assert should_process is False
    assert "SCRUM-12" in reason


def test_rank_only_change_is_ignored():
    payload = _updated_payload(
        issue_key="SCRUM-13",
        current_status="In Progress",
        changelog_items=[{"field": "Rank", "fromString": None, "toString": None}],
        changelog_id="ch-3",
    )
    should_process, reason = should_process_event(payload)
    assert should_process is False
    assert "status" in reason.lower()


def test_duplicate_delivery_second_call_is_ignored():
    payload = _updated_payload(
        issue_key="SCRUM-14",
        current_status="In Progress",
        changelog_items=[_status_changelog_item("To Do", "In Progress")],
        changelog_id="ch-4-shared",
    )
    first_should_process, _ = evaluate_webhook_event(payload)
    second_should_process, second_reason = evaluate_webhook_event(payload)

    assert first_should_process is True
    assert second_should_process is False
    assert "duplicate" in second_reason.lower()


def test_issue_created_directly_into_in_progress_is_processed():
    payload = {
        "webhookEvent": "jira:issue_created",
        "timestamp": 1720000000000,
        "issue": {
            "key": "SCRUM-15",
            "fields": {
                "summary": "Test ticket",
                "description": "Test description",
                "status": {"name": "In Progress"},
            },
        },
    }
    should_process, reason = should_process_event(payload)
    assert should_process is True
    assert "SCRUM-15" in reason


def test_issue_created_not_in_progress_is_ignored():
    payload = {
        "webhookEvent": "jira:issue_created",
        "timestamp": 1720000000000,
        "issue": {
            "key": "SCRUM-16",
            "fields": {"status": {"name": "To Do"}},
        },
    }
    should_process, _ = should_process_event(payload)
    assert should_process is False


def test_payload_missing_changelog_entirely_is_ignored_fail_safe(caplog):
    payload = {
        "webhookEvent": "jira:issue_updated",
        "timestamp": 1720000000000,
        "issue": {
            "key": "SCRUM-17",
            "fields": {
                "summary": "Test ticket",
                "description": "Test description",
                "status": {"name": "In Progress"},
            },
        },
        # No "changelog" key at all - unexpected for an issue_updated event.
    }
    should_process, reason = should_process_event(payload)
    assert should_process is False
    assert "changelog" in reason.lower()

    with caplog.at_level(logging.WARNING, logger="jira_webhook"):
        should_process, _ = evaluate_webhook_event(payload)
    assert should_process is False
    assert any("missing 'changelog' entirely" in record.message for record in caplog.records)

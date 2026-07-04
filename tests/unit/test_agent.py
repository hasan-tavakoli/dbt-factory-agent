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
from unittest.mock import MagicMock

from google.adk.agents.context import Context
from google.genai import types

from app.agent import (
    save_ticket,
    route_ticket,
    handle_config_only,
    prepare_model_builder_input,
    validate_and_push_model,
)

def test_save_ticket():
    ctx = MagicMock(spec=Context)
    ctx.state = {}
    
    input_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Please create a dbt configuration.")]
    )
    
    event = save_ticket(ctx, input_content)
    
    assert event.output == "Please create a dbt configuration."
    assert event.actions.state_delta["ticket_text"] == "Please create a dbt configuration."


def test_route_ticket():
    from unittest.mock import MagicMock
    ctx = MagicMock(spec=Context)
    classification_data = {"category": "config_only", "reason": "Only requests a config"}
    event = route_ticket(ctx, classification_data)
    
    assert event.output == classification_data
    assert event.actions.route == "ok"


def test_handle_config_only(tmp_path):
    # Change working directory to temp path for file write test
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config_data = {
            "dag_name": "test_dag",
            "schedule": "daily",
            "models": ["model_a", "model_b"],
            "owner": "team@company.com",
            "description": "Test DAG"
        }
        
        events = list(handle_config_only(config_data))
        
        assert len(events) == 2
        
        content_event = events[0]
        assert content_event.content is not None
        assert "Successfully generated dbt DAG config" in content_event.content.parts[0].text
        
        output_event = events[1]
        assert output_event.output == config_data
        
        assert os.path.exists("config.json")
        with open("config.json") as f:
            written_data = json.load(f)
        assert written_data == config_data
        
    finally:
        os.chdir(old_cwd)


def test_prepare_model_builder_input():
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "create daily active users model", "domain": "sports"}
    event = prepare_model_builder_input(ctx, {})
    assert "sports" in event.output
    assert "create daily active users model" in event.output

def test_check_sql_safety():
    from scripts.check_sql_safety import check_sql_safety
    
    # Safe queries
    is_safe, reason = check_sql_safety("SELECT * FROM raw.games")
    assert is_safe
    assert reason is None
    
    is_safe, reason = check_sql_safety("with daily_stats as (select * from games) select * from daily_stats")
    assert is_safe
    
    # Dangerous queries
    is_safe, reason = check_sql_safety("DROP TABLE raw.games")
    assert not is_safe
    assert "DROP" in reason
    
    is_safe, reason = check_sql_safety("DELETE FROM raw.games WHERE id = 1")
    assert not is_safe
    assert "DELETE" in reason
    
    is_safe, reason = check_sql_safety("CREATE OR REPLACE TABLE raw.games AS SELECT * FROM raw.old_games")
    assert not is_safe
    assert "CREATE/REPLACE" in reason


def test_validate_config_success(tmp_path):
    from unittest.mock import patch, MagicMock
    from app.agent import validate_config
    import subprocess
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = MagicMock(spec=Context)
        ctx.state = {"validation_attempts": 0}
        
        node_input = {
            "dag_configs": [
                {
                    "dag_config": {"dag_id": "my_dag", "schedule": "daily", "start_date": "2025-10-01"},
                    "job_config": {"env_variables": {"DBT_EXECUTION_PROJECT": "p", "DBT_IMPERSONATE_SERVICE_ACCOUNT": "sa", "DBT_PROJECT": "pr", "DBT_PROFILE": "prof", "DBT_LOCATION": "loc"}, "steps": [{"step_name": "run"}]}
                }
            ]
        }
        
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "OK    my_dag in dv-dev-eu"
        mock_res.stderr = ""
        
        with patch("subprocess.run", return_value=mock_res) as mock_run:
            events = list(validate_config(ctx, node_input))
            
            assert len(events) == 2
            assert "Successfully generated and validated" in events[0].content.parts[0].text
            assert events[1].actions.route == "valid"
            assert events[1].output == node_input
            
            assert os.path.exists("config.json")
            mock_run.assert_called_once()
    finally:
        os.chdir(old_cwd)


def test_validate_config_failure_retry(tmp_path):
    from unittest.mock import patch, MagicMock
    from app.agent import validate_config
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = MagicMock(spec=Context)
        ctx.state = {"validation_attempts": 0}
        
        node_input = {"test": "data"}
        
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stdout = "  - [dag_configs -> 0 -> job_config -> steps] Field required"
        mock_res.stderr = "Validation failed. Fix the fields listed above."
        
        with patch("subprocess.run", return_value=mock_res) as mock_run:
            events = list(validate_config(ctx, node_input))
            
            assert len(events) == 1
            assert events[0].actions.route == "retry"
            assert events[0].actions.state_delta["validation_attempts"] == 1
            assert "Field required" in events[0].actions.state_delta["validation_feedback"]
            
            assert os.path.exists("config.json")
            mock_run.assert_called_once()
    finally:
        os.chdir(old_cwd)


def test_validate_config_failure_max_attempts(tmp_path):
    from unittest.mock import patch, MagicMock
    from app.agent import validate_config
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = MagicMock(spec=Context)
        ctx.state = {"validation_attempts": 3}  # 3 attempts already made, this is the 4th
        
        node_input = {"test": "data"}
        
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stdout = "  - [dag_configs -> 0 -> job_config -> steps] Field required"
        mock_res.stderr = "Validation failed. Fix the fields listed above."
        
        with patch("subprocess.run", return_value=mock_res) as mock_run:
            events = list(validate_config(ctx, node_input))
            
            assert len(events) == 2
            assert "Validation failed after 4 attempts" in events[0].content.parts[0].text
            assert events[1].actions.route == "needs_human"
            
            assert os.path.exists("config.json")
            mock_run.assert_called_once()
    finally:
        os.chdir(old_cwd)


def test_check_critical_fields_success():
    from unittest.mock import MagicMock
    from app.agent import check_critical_fields
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: stage"}
    node_input = {
        "dag_configs": [
            {
                "dag_config": {"dag_id": "my_dag", "schedule": "daily", "start_date": "2025-10-01"},
                "job_config": {
                    "env_variables": {
                        "DBT_EXECUTION_PROJECT": "p",
                        "DBT_IMPERSONATE_SERVICE_ACCOUNT": "sa",
                        "DBT_PROJECT": "pr",
                        "DBT_PROFILE": "prof",
                        "DBT_LOCATION": "loc"
                    },
                    "steps": [{"step_name": "run"}]
                }
            }
        ]
    }
    
    event = check_critical_fields(ctx, node_input)
    assert event.actions.route == "ok"
    assert "Resolved target path: dv-platform-config/dv-stage-eu/sports/my-dag/config.json" in event.content.parts[0].text
    assert event.output == node_input


def test_check_critical_fields_missing():
    from unittest.mock import MagicMock
    from app.agent import check_critical_fields
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: stage"}
    # Missing DBT_IMPERSONATE_SERVICE_ACCOUNT
    node_input = {
        "dag_configs": [
            {
                "dag_config": {"dag_id": "my_dag", "schedule": "daily", "start_date": "2025-10-01"},
                "job_config": {
                    "env_variables": {
                        "DBT_EXECUTION_PROJECT": "p",
                        "DBT_PROJECT": "pr",
                        "DBT_PROFILE": "prof",
                        "DBT_LOCATION": "loc"
                    },
                    "steps": [{"step_name": "run"}]
                }
            }
        ]
    }
    
    event = check_critical_fields(ctx, node_input)
    assert event.actions.route == "needs_human"
    assert "Critical fields/metadata are missing" in event.output
    assert "DBT_IMPERSONATE_SERVICE_ACCOUNT" in event.output


def test_check_critical_fields_missing_env():
    from unittest.mock import MagicMock
    from app.agent import check_critical_fields
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "DAG id my_dag..."}
    node_input = {
        "dag_configs": [
            {
                "dag_config": {"dag_id": "my_dag", "schedule": "daily", "start_date": "2025-10-01"},
                "job_config": {
                    "env_variables": {
                        "DBT_EXECUTION_PROJECT": "p",
                        "DBT_IMPERSONATE_SERVICE_ACCOUNT": "sa",
                        "DBT_PROJECT": "pr",
                        "DBT_PROFILE": "prof",
                        "DBT_LOCATION": "loc"
                    },
                    "steps": [{"step_name": "run"}]
                }
            }
        ]
    }
    
    event = check_critical_fields(ctx, node_input)
    assert event.actions.route == "needs_human"
    assert "environment" in event.output
    assert "domain" in event.output


def test_check_critical_fields_prod_guard():
    from unittest.mock import MagicMock
    from app.agent import check_critical_fields
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: prod"}
    node_input = {
        "dag_configs": [
            {
                "dag_config": {"dag_id": "my_dag", "schedule": "daily", "start_date": "2025-10-01"},
                "job_config": {
                    "env_variables": {
                        "DBT_EXECUTION_PROJECT": "p",
                        "DBT_IMPERSONATE_SERVICE_ACCOUNT": "sa",
                        "DBT_PROJECT": "pr",
                        "DBT_PROFILE": "prof",
                        "DBT_LOCATION": "loc"
                    },
                    "steps": [{"step_name": "run"}]
                }
            }
        ]
    }
    
    event = check_critical_fields(ctx, node_input)
    assert event.actions.route == "needs_human"
    assert "production must be handled by a human" in event.output


def test_handle_needs_human(tmp_path):
    from unittest.mock import MagicMock
    from app.agent import handle_needs_human
    import json
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = MagicMock(spec=Context)
        ctx.state = {
            "ticket_text": "Domain: sports...",
            "missing_critical_fields": ["DBT_IMPERSONATE_SERVICE_ACCOUNT"]
        }
        
        events = list(handle_needs_human(ctx, "Some validation error"))
        
        assert len(events) == 2
        assert "Ticket routed to human review" in events[0].content.parts[0].text
        
        # Verify JSONL log file is created
        assert os.path.exists("needs_human_queue.jsonl")
        with open("needs_human_queue.jsonl", "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["ticket"] == "Domain: sports..."
        assert record["missing_critical_fields"] == ["DBT_IMPERSONATE_SERVICE_ACCOUNT"]
        assert record["reason"] == "Some validation error"
    finally:
        os.chdir(old_cwd)


def test_check_domain_exact_success():
    from unittest.mock import MagicMock
    from app.agent import check_domain_exact
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: stage"}
    
    event = check_domain_exact(ctx, "input_data")
    assert event.actions.route == "ok"
    assert event.actions.state_delta["domain"] == "sports"
    assert event.output == "input_data"


def test_check_domain_exact_failure():
    from unittest.mock import MagicMock
    from app.agent import check_domain_exact
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sprot, Environment: stage"}
    
    event = check_domain_exact(ctx, "input_data")
    assert event.actions.route == "typo_check"
    assert event.output == {"domain": "sprot"}


def test_validate_domain_typo_result_success():
    from unittest.mock import MagicMock
    from app.agent import validate_domain_typo_result
    from google.adk.events.request_input import RequestInput
    
    ctx = MagicMock(spec=Context)
    ctx.node_path = "path"
    node_input = {"suggested_domain": "sports", "reason": "Close to sprot"}
    
    events = list(validate_domain_typo_result(ctx, node_input))
    assert len(events) == 2
    assert isinstance(events[0], RequestInput)
    assert events[0].message == "Did you mean 'sports'?"
    assert events[1].actions.state_delta["suggested_domain"] == "sports"


def test_validate_domain_typo_result_unrelated(tmp_path):
    from unittest.mock import MagicMock
    from app.agent import validate_domain_typo_result
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = MagicMock(spec=Context)
        ctx.state = {"ticket_text": "Domain: banana..."}
        node_input = {"suggested_domain": "", "reason": "Not related"}
        
        events = list(validate_domain_typo_result(ctx, node_input))
        assert len(events) == 2
        assert "Domain validation failed" in events[0].content.parts[0].text
        assert events[1].actions.route == "stop"
        
        # Verify JSONL log file is created
        assert os.path.exists("wrong_domain_queue.jsonl")
    finally:
        os.chdir(old_cwd)


def test_handle_domain_confirmation_yes():
    from unittest.mock import MagicMock
    from app.agent import handle_domain_confirmation
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"suggested_domain": "sports"}
    
    event = handle_domain_confirmation(ctx, "yes")
    assert event.actions.route == "ok"
    assert event.actions.state_delta["domain"] == "sports"


def test_handle_domain_confirmation_no():
    from unittest.mock import MagicMock
    from app.agent import handle_domain_confirmation
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"suggested_domain": "sports"}
    
    event = handle_domain_confirmation(ctx, "no")
    assert event.actions.route == "stop"
    assert "User declined" in event.content.parts[0].text


def test_handle_domain_confirmation_override():
    from unittest.mock import MagicMock
    from app.agent import handle_domain_confirmation
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"suggested_domain": "sports"}
    
    event = handle_domain_confirmation(ctx, "wallet")
    assert event.actions.route == "ok"
    assert event.actions.state_delta["domain"] == "wallet"
    assert event.output == "wallet"


def test_handle_domain_confirmation_unrecognized():
    from unittest.mock import MagicMock
    from app.agent import handle_domain_confirmation
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"suggested_domain": "sports"}
    
    event = handle_domain_confirmation(ctx, "banana")
    assert event.actions.route == "stop"
    assert "Unrecognized domain response" in event.content.parts[0].text


def test_check_env_exact_success():
    from unittest.mock import MagicMock
    from app.agent import check_env_exact
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: stage"}
    
    event = check_env_exact(ctx, "input_data")
    assert event.actions.route == "ok"
    assert event.actions.state_delta["environment"] == "stage"
    assert event.output == "input_data"


def test_check_env_exact_failure():
    from unittest.mock import MagicMock
    from app.agent import check_env_exact
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: stag"}
    
    event = check_env_exact(ctx, "input_data")
    assert event.actions.route == "typo_check"
    assert event.output == {"env": "stag"}


def test_check_env_exact_prod():
    from unittest.mock import MagicMock
    from app.agent import check_env_exact
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Domain: sports, Environment: prod"}
    
    event = check_env_exact(ctx, "input_data")
    assert event.actions.route == "prod_guard"
    assert "production must be handled by a human" in event.output


def test_validate_env_typo_result_success():
    from unittest.mock import MagicMock
    from app.agent import validate_env_typo_result
    from google.adk.events.request_input import RequestInput
    
    ctx = MagicMock(spec=Context)
    ctx.node_path = "path"
    node_input = {"suggested_env": "stage", "reason": "Close to stag"}
    
    events = list(validate_env_typo_result(ctx, node_input))
    assert len(events) == 2
    assert isinstance(events[0], RequestInput)
    assert events[0].message == "Did you mean 'stage'?"
    assert events[1].actions.state_delta["suggested_env"] == "stage"


def test_validate_env_typo_result_unrelated(tmp_path):
    from unittest.mock import MagicMock
    from app.agent import validate_env_typo_result
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = MagicMock(spec=Context)
        ctx.state = {"ticket_text": "Environment: xyz..."}
        node_input = {"suggested_env": "", "reason": "Not related"}
        
        events = list(validate_env_typo_result(ctx, node_input))
        assert len(events) == 2
        assert "Environment validation failed" in events[0].content.parts[0].text
        assert events[1].actions.route == "stop"
        
        assert os.path.exists("wrong_domain_queue.jsonl")
    finally:
        os.chdir(old_cwd)


def test_handle_env_confirmation_yes():
    from unittest.mock import MagicMock
    from app.agent import handle_env_confirmation
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"suggested_env": "stage"}
    
    event = handle_env_confirmation(ctx, "yes")
    assert event.actions.route == "ok"
    assert event.actions.state_delta["environment"] == "stage"


def test_handle_env_confirmation_no():
    from unittest.mock import MagicMock
    from app.agent import handle_env_confirmation
    
    ctx = MagicMock(spec=Context)
    ctx.state = {"suggested_env": "stage"}
    
    event = handle_env_confirmation(ctx, "no")
    assert event.actions.route == "stop"
    assert "User declined" in event.content.parts[0].text


def test_check_sql_safety_early_safe():
    from unittest.mock import MagicMock
    from app.agent import check_sql_safety_early
    
    ctx = MagicMock(spec=Context)
    node_input = "Here is the SQL:\n```sql\nselect * from raw.bets;\n```"
    
    events = list(check_sql_safety_early(ctx, node_input))
    assert len(events) == 1
    assert events[0].actions.route == "safe"
    assert events[0].actions.state_delta["generated_text"] == node_input


def test_check_sql_safety_early_unsafe():
    from unittest.mock import MagicMock
    from app.agent import check_sql_safety_early
    
    ctx = MagicMock(spec=Context)
    node_input = "Here is the SQL:\n```sql\ndrop table raw.bets;\n```"
    
    events = list(check_sql_safety_early(ctx, node_input))
    assert len(events) == 2
    assert "SQL safety check rejected" in events[0].content.parts[0].text
    assert events[1].actions.route == "unsafe"


def test_reject_ticket_non_jira(monkeypatch):
    from unittest.mock import MagicMock
    from app.agent import reject_ticket

    # Mock Pub/Sub PublisherClient
    mock_publisher = MagicMock()
    mock_publisher.topic_path.return_value = "projects/ht-project-500813/topics/dv-rejected-tickets"
    mock_future = MagicMock()
    mock_future.result.return_value = "msg-123"
    mock_publisher.publish.return_value = mock_future
    
    monkeypatch.setattr("google.cloud.pubsub_v1.PublisherClient", lambda: mock_publisher)

    # Mock httpx.post to make sure no HTTP call is made for non-jira user
    mock_post = MagicMock()
    monkeypatch.setattr("httpx.post", mock_post)

    reject_ticket(
        reason_category="unsafe_sql",
        reason_text="DROP is not allowed",
        ticket="Some ticket text",
        user_id="some-user-123"
    )

    # Verify Pub/Sub was called
    mock_publisher.publish.assert_called_once()
    # Verify httpx.post was NOT called because user_id does not start with "jira-"
    mock_post.assert_not_called()


def test_reject_ticket_jira(monkeypatch):
    from unittest.mock import MagicMock
    from app.agent import reject_ticket

    # Mock Pub/Sub PublisherClient
    mock_publisher = MagicMock()
    mock_publisher.topic_path.return_value = "projects/ht-project-500813/topics/dv-rejected-tickets"
    mock_future = MagicMock()
    mock_future.result.return_value = "msg-123"
    mock_publisher.publish.return_value = mock_future
    
    monkeypatch.setattr("google.cloud.pubsub_v1.PublisherClient", lambda: mock_publisher)

    # Mock httpx.post to simulate successful Jira comment (status 201)
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr("httpx.post", mock_post)

    # Set dummy env vars for credentials
    monkeypatch.setenv("JIRA_EMAIL", "test@test.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "dummy-token")
    monkeypatch.setenv("JIRA_BASE_URL", "https://hassan-t.atlassian.net")

    reject_ticket(
        reason_category="invalid_domain",
        reason_text="Unknown domain: marketing",
        ticket="Some ticket text",
        user_id="jira-SPORT-101"
    )

    # Verify Pub/Sub was called
    mock_publisher.publish.assert_called_once()
    # Verify httpx.post was called to comment on Jira
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://hassan-t.atlassian.net/rest/api/3/issue/SPORT-101/comment"
    assert kwargs["auth"] == ("test@test.com", "dummy-token")


def test_prepare_model_builder_input_stashes_intent_payload_from_model_instance():
    from app.agent import IntentPayload

    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Add a model", "domain": "sports"}

    payload = IntentPayload(
        service_account="sa@project.iam.gserviceaccount.com",
        execution_project="proj-exec",
        target_project="proj-target",
        dag_id="dv_sports_elt",
        schedule="0 6 * * *",
        config_intent=None,
    )

    event = prepare_model_builder_input(ctx, payload)

    assert event.actions.state_delta["model_intent_payload"] == payload.model_dump()


def test_prepare_model_builder_input_stashes_dict_as_is():
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Add a model", "domain": "sports"}

    payload_dict = {"service_account": "sa@x", "schedule": "0 6 * * *"}
    event = prepare_model_builder_input(ctx, payload_dict)

    assert event.actions.state_delta["model_intent_payload"] == payload_dict


def test_prepare_model_builder_input_falls_back_to_empty_dict_for_unknown_shape():
    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Add a model", "domain": "sports"}

    event = prepare_model_builder_input(ctx, "not a payload")

    assert event.actions.state_delta["model_intent_payload"] == {}


def test_check_sql_safety_early_safe_outputs_stashed_intent_payload_not_raw_text():
    from app.agent import check_sql_safety_early

    ctx = MagicMock(spec=Context)
    ctx.state = {"model_intent_payload": {"service_account": "sa@x", "schedule": "0 6 * * *"}}

    node_input = "Here is the SQL:\n```sql\nselect * from raw.bets;\n```"
    events = list(check_sql_safety_early(ctx, node_input))

    safe_event = events[-1]
    assert safe_event.actions.route == "safe"
    assert safe_event.output == {"service_account": "sa@x", "schedule": "0 6 * * *"}
    assert not isinstance(safe_event.output, str)


def test_model_only_handoff_wiring_carries_intent_payload_through_sql_generation():
    """
    Regression test for the exact production bug: check_sql_safety_early's
    'safe' route feeds validate_config_only_payload, which requires an
    IntentPayload-coercible value (a dict or IntentPayload instance) as
    node_input - NOT the raw SQL/YAML markdown string model_builder produces.
    This chains prepare_model_builder_input -> check_sql_safety_early exactly
    as the real graph does, applying prepare_model_builder_input's state
    delta the way the ADK engine would before check_sql_safety_early runs.
    """
    from app.agent import IntentPayload, check_sql_safety_early

    ctx = MagicMock(spec=Context)
    ctx.state = {"ticket_text": "Add a model for sports", "domain": "sports"}

    extracted_payload = IntentPayload(
        service_account="sa@project.iam.gserviceaccount.com",
        execution_project="proj-exec",
        target_project="proj-target",
        dag_id="dv_sports_elt",
        schedule="0 6 * * *",
        config_intent=None,
    )

    # Step 1: model_intent_extractor -> prepare_model_builder_input
    prep_event = prepare_model_builder_input(ctx, extracted_payload)
    # Simulate the ADK engine merging the state delta before the next node runs.
    ctx.state.update(prep_event.actions.state_delta)

    # Step 2: model_builder's generated text -> check_sql_safety_early
    model_builder_output = (
        "```sql\n-- filepath: dbt/models/public/daily_active_users.sql\n"
        "select user_id, count(*) from raw.events group by user_id\n```\n"
        "```yaml\n# filepath: dbt/models/public/_schema.yml\nversion: 2\n```"
    )
    events = list(check_sql_safety_early(ctx, model_builder_output))
    safe_event = events[-1]

    assert safe_event.actions.route == "safe"
    # This is the exact assertion that would have failed before the fix:
    # output used to be the raw markdown string, which crashes
    # validate_config_only_payload's IntentPayload-typed node_input.
    assert isinstance(safe_event.output, dict)
    assert safe_event.output == extracted_payload.model_dump()

    # Confirm it's actually coercible to IntentPayload, matching what
    # validate_config_only_payload's type hint requires.
    coerced = IntentPayload(**safe_event.output)
    assert coerced.service_account == "sa@project.iam.gserviceaccount.com"
    assert coerced.schedule == "0 6 * * *"


def test_validate_and_push_model_sets_git_identity_before_commit(monkeypatch):
    """
    Regression test for "Author identity unknown" (git exit code 128): the
    Agent Runtime container has no global git user.name/user.email, so
    `git commit` in the fresh temp clone used to fail outright. This asserts
    the local (non-global) identity is configured, and configured strictly
    before the commit call - not just present anywhere in the call list.
    """
    from unittest.mock import patch, MagicMock
    from app.agent import validate_and_push_model

    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")

    ctx = MagicMock(spec=Context)
    ctx.state = {
        "generated_text": (
            "```sql\n-- filepath: dbt/models/public/daily_active_users.sql\n"
            "select 1\n```\n"
            "```yaml\n# filepath: dbt/models/public/_schema.yml\nversion: 2\n```"
        ),
        "agent_metadata": None,
    }

    success = MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", return_value=success) as mock_run:
        events = list(validate_and_push_model(ctx, "unused"))

    commands = [call.args[0] for call in mock_run.call_args_list]

    email_idx = commands.index(["git", "config", "user.email", "agent@dbt-factory-agent.local"])
    name_idx = commands.index(["git", "config", "user.name", "dbt-factory-agent"])
    commit_idx = next(i for i, c in enumerate(commands) if c[:2] == ["git", "commit"])

    assert email_idx < commit_idx
    assert name_idx < commit_idx
    assert events[-1].actions.route == "ok"


def test_validate_and_push_model_surfaces_real_git_stderr_on_commit_failure(monkeypatch):
    """
    Git subprocess failures used to be swallowed: `check=True` without
    `capture_output=True` means CalledProcessError.stderr is None, so the
    real git error text never reached the logs/error message. This asserts
    a failing `git commit` now surfaces its actual stderr.
    """
    from unittest.mock import patch, MagicMock
    from app.agent import validate_and_push_model

    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")

    ctx = MagicMock(spec=Context)
    ctx.state = {
        "generated_text": (
            "```sql\n-- filepath: dbt/models/public/daily_active_users.sql\n"
            "select 1\n```\n"
        ),
        "agent_metadata": None,
    }

    success = MagicMock(returncode=0, stdout="", stderr="")
    commit_failure = MagicMock(
        returncode=128,
        stdout="",
        stderr="Author identity unknown\n\nfatal: unable to auto-detect email address",
    )

    def run_side_effect(cmd, **kwargs):
        if cmd[:2] == ["git", "commit"]:
            return commit_failure
        return success

    with patch("subprocess.run", side_effect=run_side_effect):
        events = list(validate_and_push_model(ctx, "unused"))

    assert events[-1].actions.route == "needs_human"
    assert "Author identity unknown" in events[0].content.parts[0].text

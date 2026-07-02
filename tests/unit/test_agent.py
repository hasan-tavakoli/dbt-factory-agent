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
    handle_model_only,
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


def test_handle_model_only():
    events = list(handle_model_only({}))
    assert len(events) == 2
    assert events[0].content.parts[0].text == "not implemented yet"
    assert events[1].output == "not implemented yet"


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
        mock_res.stdout = "OK    my_dag in leo-dev-eu"
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
    assert "Resolved target path: leo-stage-eu/sports/my-dag/config.json" in event.content.parts[0].text
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

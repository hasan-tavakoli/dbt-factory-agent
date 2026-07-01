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
    classification_data = {"category": "config_only", "reason": "Only requests a config"}
    event = route_ticket(classification_data)
    
    assert event.output == classification_data
    assert event.actions.route == "config_only"


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

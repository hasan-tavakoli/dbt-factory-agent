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
Covers jira_webhook.main.adf_to_plain_text: Jira Cloud sends `description` as
an ADF document as soon as it has any rich formatting, which used to get
naively stringified and silently break downstream regex parsing of labeled
fields like "Domain: sprot".
"""

from jira_webhook.main import adf_to_plain_text
from scripts.check_required_fields import parse_env_and_domain

# "Domain:" is bold, "sprot" is a separate plain-text run in the same
# paragraph — exactly the shape that broke parse_env_and_domain before this
# fix (bold label + plain value split into two ADF text nodes).
ADF_WITH_SPLIT_BOLD_LABEL = {
    "type": "doc",
    "version": 1,
    "content": [
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "Domain: ", "marks": [{"type": "strong"}]},
                {"type": "text", "text": "sprot"},
            ],
        },
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "Environment: stage"},
            ],
        },
    ],
}


def test_adf_with_split_bold_label_yields_contiguous_domain_field() -> None:
    plain_text = adf_to_plain_text(ADF_WITH_SPLIT_BOLD_LABEL)

    assert "Domain: sprot" in plain_text
    assert "Environment: stage" in plain_text

    env, domain = parse_env_and_domain(plain_text)
    assert domain == "sprot"
    assert env == "stage"


def test_adf_hard_break_becomes_newline() -> None:
    adf = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Domain: sprot"},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "Environment: stage"},
                ],
            }
        ],
    }
    plain_text = adf_to_plain_text(adf)
    assert plain_text == "Domain: sprot\nEnvironment: stage"


def test_none_description_returns_empty_string() -> None:
    assert adf_to_plain_text(None) == ""


def test_plain_string_description_passes_through_unchanged() -> None:
    assert adf_to_plain_text("Domain: sprot\nEnvironment: stage") == "Domain: sprot\nEnvironment: stage"


def test_unexpected_shape_falls_back_to_str_without_raising() -> None:
    # Not a dict/str/None — must not raise, must degrade gracefully.
    assert adf_to_plain_text(12345) == "12345"


def test_malformed_adf_dict_falls_back_without_raising() -> None:
    # A non-dict, non-string item inside "content" — must not crash, must
    # coerce to text instead.
    malformed = {"type": "doc", "content": [42, {"type": "text", "text": "ok"}]}
    result = adf_to_plain_text(malformed)
    assert isinstance(result, str)
    assert "42" in result
    assert "ok" in result

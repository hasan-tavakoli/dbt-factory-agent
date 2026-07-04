import json
import logging
import os
from collections import deque
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from google.cloud import pubsub_v1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira_webhook")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "ht-project-500813")
TOPIC_ID = os.getenv("PUBSUB_TOPIC", "dv-jira-tickets")

app = FastAPI(title="Jira Webhook Handler")

# Initialize Pub/Sub Publisher Client
# Note: Google credentials will be automatically resolved from the environment / service account
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

def _adf_node_to_text(node: Any) -> str:
    """Recursively renders a single ADF node (and its children) to plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return str(node)

    node_type = node.get("type")

    if node_type == "hardBreak":
        return "\n"

    parts = []
    text = node.get("text")
    if text:
        parts.append(text)

    for child in node.get("content") or []:
        parts.append(_adf_node_to_text(child))

    rendered = "".join(parts)

    # Block-level nodes each end their own line, so sibling paragraphs/list
    # items don't get smashed together with whatever follows them.
    if node_type in ("paragraph", "heading", "listItem", "blockquote", "codeBlock", "rule"):
        rendered += "\n"

    return rendered


def adf_to_plain_text(description: Any) -> str:
    """
    Converts a Jira description field to plain text.

    Handles the shapes Jira actually sends:
    - None -> ""
    - str -> returned as-is (API v2 / already-plain text)
    - dict (ADF document) -> recursively walked, concatenating every
      {"type": "text", "text": ...} node regardless of which marks
      (bold/italic/etc.) split it into separate runs, and turning
      paragraph/heading/hardBreak boundaries into newlines.

    Anything else falls back to str(description) rather than raising, since
    this only feeds a best-effort ticket-text string for the LLM.
    """
    if description is None:
        return ""
    if isinstance(description, str):
        return description
    if isinstance(description, dict):
        try:
            return _adf_node_to_text(description).strip()
        except Exception as e:
            logger.warning(f"Failed to parse ADF description, falling back to str(): {e}")
            return str(description)
    return str(description)


# ── Transition check + dedup ────────────────────────────────────────────────
#
# The webhook used to fire whenever an issue's CURRENT status was "In
# Progress", regardless of what actually changed in that event (a STATE
# check). That caused two bugs in production:
#   1. Any incidental touch on an already-"In Progress" issue (a sibling
#      rank change, or Jira/Automation re-delivering the same event) got
#      reprocessed as if it were a fresh transition.
#   2. The orchestrator's own rejection COMMENT on a Jira issue triggers an
#      "issue updated" event, which would loop back through that same state
#      check forever: comment -> webhook fires -> reject -> comment -> ...
#
# The checks below only act on an event if THAT EVENT'S OWN changelog shows
# the status field moving into "In Progress" (a TRANSITION check) - there is
# no fallback to the issue's current status.

_RECENT_TRANSITION_KEYS_MAXLEN = 200
_recent_transition_keys: deque = deque(maxlen=_RECENT_TRANSITION_KEYS_MAXLEN)

# Comment events never carry a status changelog by design - that's expected,
# not a sign the Automation rule dropped something.
_COMMENT_EVENT_TYPES = {"comment_created", "comment_updated", "comment_deleted"}


def _get_status_changelog_item(changelog: dict) -> dict | None:
    """Returns the changelog item for the 'status' field, if present."""
    if not isinstance(changelog, dict):
        return None
    for item in changelog.get("items") or []:
        if isinstance(item, dict) and item.get("field") == "status":
            return item
    return None


def _payload_missing_changelog(payload: dict) -> bool:
    """
    True only when a changelog is genuinely expected but absent. A
    "jira:issue_created" event never has one, and comment events never carry
    a status changelog either - both are expected, not a misconfiguration.
    Anything else missing "changelog" entirely is an unexpected payload
    shape, worth flagging: this webhook is called by a Jira Automation
    rule's "Send web request" action (not a native webhook - see the
    ?triggeredByUser= query param), so we can't yet be 100% sure it forwards
    Jira's native changelog verbatim.
    """
    webhook_event = payload.get("webhookEvent")
    if webhook_event == "jira:issue_created" or webhook_event in _COMMENT_EVENT_TYPES:
        return False
    return "changelog" not in payload


def should_process_event(payload: dict) -> tuple[bool, str]:
    """
    Decides whether this webhook delivery represents a genuine transition
    INTO "In Progress" that the orchestrator should act on.

    This is a TRANSITION check, not a state check: an event is only acted on
    if its own changelog shows the status field moving into "In Progress".
    Anything without a status changelog item - comment-only edits, rank-only
    touches, sibling-issue touch events, and critically the agent's own
    rejection comment - is ignored outright, with no fallback to inspecting
    the issue's current status.
    """
    issue = payload.get("issue") or {}
    issue_key = issue.get("key", "UNKNOWN")
    webhook_event = payload.get("webhookEvent")

    # Brand-new issues have no changelog (nothing "changed" from a prior
    # state) - only act if it was created directly into "In Progress".
    if webhook_event == "jira:issue_created":
        fields = issue.get("fields") or {}
        status_name = (fields.get("status") or {}).get("name") or ""
        if status_name.lower() == "in progress":
            return True, f"{issue_key}: created directly into In Progress"
        return False, f"{issue_key}: created with status '{status_name}', not In Progress"

    if _payload_missing_changelog(payload):
        return False, (
            f"{issue_key}: no 'changelog' key in payload at all "
            f"(webhookEvent={webhook_event!r}) - ignoring to fail safe"
        )

    changelog = payload.get("changelog") or {}
    status_item = _get_status_changelog_item(changelog)

    if status_item is None:
        # Comment-only edits, rank-only touches, sibling-issue touch events,
        # and the agent's own rejection comment all land here.
        return False, f"{issue_key}: event's changelog has no 'status' field change - ignoring"

    from_string = (status_item.get("fromString") or "").strip().lower()
    to_string = (status_item.get("toString") or "").strip().lower()

    if to_string == "in progress" and from_string != "in progress":
        return True, (
            f"{issue_key}: status transitioned "
            f"'{status_item.get('fromString')}' -> '{status_item.get('toString')}'"
        )

    return False, (
        f"{issue_key}: status changelog present but not a transition into In "
        f"Progress (from='{status_item.get('fromString')}', "
        f"to='{status_item.get('toString')}')"
    )


def _dedup_key(payload: dict, issue_key: str) -> str:
    """
    Best-effort key identifying "this specific real change event", so the
    millisecond-apart double deliveries seen in production logs (same issue,
    same changelog id, delivered twice) can be told apart from two genuinely
    different transitions on the same issue.
    """
    changelog = payload.get("changelog") or {}
    changelog_id = changelog.get("id")
    if changelog_id:
        return f"{issue_key}:{changelog_id}"
    # jira:issue_created has no changelog id - fall back to event + timestamp.
    return f"{issue_key}:{payload.get('webhookEvent', '')}:{payload.get('timestamp', '')}"


def _is_duplicate_delivery(dedup_key: str) -> bool:
    """
    Best-effort in-memory de-dup for repeat deliveries of the same change
    event. Doesn't survive a restart and isn't shared across Cloud Run
    instances, but catches the warm-instance, millisecond-apart duplicates
    that are the actual observed failure mode. Deliberately simple - no
    external state store.
    """
    if dedup_key in _recent_transition_keys:
        return True
    _recent_transition_keys.append(dedup_key)
    return False


def evaluate_webhook_event(payload: dict) -> tuple[bool, str]:
    """
    Full decision for a webhook delivery: should it result in a ticket being
    published to Pub/Sub? Combines the transition check with duplicate-
    delivery de-dup. This is what the route handler calls.
    """
    issue = payload.get("issue") or {}
    issue_key = issue.get("key", "UNKNOWN")

    if _payload_missing_changelog(payload):
        logger.warning(
            f"Payload for issue key={issue_key} is missing 'changelog' entirely "
            f"(webhookEvent={payload.get('webhookEvent')!r}). This is unexpected "
            f"for an issue-update-style event delivered via the Jira Automation "
            f"rule - verify its 'Send web request' body actually forwards Jira's "
            f"native changelog. Ignoring this event to fail safe."
        )

    should_process, reason = should_process_event(payload)
    if not should_process:
        return False, reason

    dedup_key = _dedup_key(payload, issue_key)
    if _is_duplicate_delivery(dedup_key):
        return False, f"{issue_key}: duplicate delivery ignored (dedup key={dedup_key})"

    return True, reason


@app.post("/jira-webhook")
async def handle_jira_webhook(request: Request):
    """Receives Jira issue transition/created webhook and publishes to Pub/Sub."""
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON payload"})

    # TEMP DEBUG (remove once verified on a real transition): confirm the
    # Automation rule's "Send web request" body actually forwards Jira's
    # native changelog. Logs only the top-level shape, never full
    # contents/secrets.
    logger.info(
        f"[TEMP DEBUG] Incoming payload top-level keys: {sorted(payload.keys())}, "
        f"has_changelog={'changelog' in payload}"
    )

    # 1. Decide whether this event is a genuine transition into "In
    # Progress" (see should_process_event/evaluate_webhook_event above).
    issue = payload.get("issue", {})
    issue_key = issue.get("key", "UNKNOWN")

    should_process, reason = evaluate_webhook_event(payload)
    logger.info(f"Transition check for issue key={issue_key}: should_process={should_process} ({reason})")

    if not should_process:
        return {"status": "ignored", "message": reason}

    # 2. Extract summary, description
    fields = issue.get("fields", {})
    summary = fields.get("summary", "") or ""
    description = fields.get("description")

    # 3. Format message text (summary + newline + description).
    # Jira Cloud (API v3) sends `description` as an ADF document, not a plain
    # string, as soon as it has any rich formatting — flatten it first so
    # labeled fields like "Domain: sprot" survive regardless of how they were
    # styled in the Jira editor.
    plain_description = adf_to_plain_text(description)
    ticket_text = f"{summary}\n{plain_description}"

    # 4. Formulate the exact streamQuery payload structure
    orchestrator_payload = {
        "class_method": "stream_query",
        "input": {
            "message": ticket_text,
            "user_id": f"jira-{issue_key}"
        }
    }

    # 5. Publish payload to Pub/Sub
    try:
        data_str = json.dumps(orchestrator_payload)
        data_bytes = data_str.encode("utf-8")
        
        logger.info(f"Publishing payload for {issue_key} to topic {topic_path}...")
        future = publisher.publish(topic_path, data=data_bytes)
        message_id = future.result()
        logger.info(f"Successfully published {issue_key}. Message ID: {message_id}")
        return {"status": "success", "message_id": message_id}
    except Exception as e:
        logger.error(f"Failed to publish message to Pub/Sub for {issue_key}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"Failed to publish payload: {str(e)}"}
        )

@app.get("/")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

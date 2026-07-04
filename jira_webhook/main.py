import json
import logging
import os
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


@app.post("/jira-webhook")
async def handle_jira_webhook(request: Request):
    """Receives Jira issue transition/created webhook and publishes to Pub/Sub."""
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON payload"})

    # 1. Inspect status to decide whether to process
    issue = payload.get("issue", {})
    fields = issue.get("fields", {})
    issue_key = issue.get("key", "UNKNOWN")
    
    status_name = None
    if isinstance(fields, dict):
        status_info = fields.get("status", {})
        if isinstance(status_info, dict):
            status_name = status_info.get("name")
            
    logger.info(f"Received webhook for issue key={issue_key}, status={status_name}")

    # Process if status is 'In Progress', or if status is not present at all.
    # If status is present but is not 'In Progress', ignore the event.
    if status_name is not None and status_name.lower() != "in progress":
        msg = f"Ignoring issue {issue_key}: status is '{status_name}' (expected 'In Progress')."
        logger.info(msg)
        return {"status": "ignored", "message": msg}

    # 2. Extract issue key, summary, description
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

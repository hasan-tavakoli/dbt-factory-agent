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
    description = fields.get("description", "") or ""

    # 3. Format message text (summary + newline + description)
    ticket_text = f"{summary}\n{description}"

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

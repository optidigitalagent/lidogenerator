"""Post-finalization facade that keeps Opti failures outside discovery/export."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import db
from integrations import opti_outbox
from integrations.opti_contract import build_payload

log = logging.getLogger("lead_hunter.opti_bridge")


async def finalize_completed_task(task_id: int) -> str:
    """Enqueue the persisted final set once, then make one best-effort delivery."""
    task_row = db.get_task(task_id)
    if task_row is None:
        return "Opti sync pending — /sync"
    task = dict(task_row)
    businesses = db.get_businesses_for_bridge(task_id)
    if task.get("status") != "done" or not businesses:
        return ""
    try:
        payload = build_payload(
            task,
            businesses,
            generated_at=datetime.now(timezone.utc),
        )
        opti_outbox.enqueue(payload)
        await opti_outbox.deliver_due(external_batch_id=payload["externalBatchId"], limit=1)
        row = opti_outbox.get_by_batch(payload["externalBatchId"])
        if row and row["status"] == "SENT" and row["responseSummaryJson"]:
            import json

            response = json.loads(row["responseSummaryJson"])
            review = response["duplicateCount"] + response["rejectedCount"]
            return (
                f"Opti: {response['createdCount']} created, "
                f"{response['updatedCount']} updated, {review} review required"
            )
    except Exception:
        log.exception("Opti bridge finalization failed for task %s", task_id)
    return "Opti sync pending — /sync"

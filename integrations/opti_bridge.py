"""Post-finalization facade that keeps Opti failures outside discovery/export."""

from __future__ import annotations

import logging

import db
from integrations import opti_outbox
from integrations.opti_contract import SCHEMA_VERSION, build_payload

log = logging.getLogger("lead_hunter.opti_bridge")


def reconcile_completed_tasks(limit: int = 100) -> dict[str, int]:
    """Enqueue missing eligible completed tasks without making HTTP calls."""
    summary = {"examined": 0, "enqueued": 0, "errors": 0}
    for task_row in db.get_completed_tasks_for_opti_reconciliation(limit=limit):
        summary["examined"] += 1
        task = dict(task_row)
        try:
            businesses = db.get_businesses_for_bridge(task["id"])
            payload = build_payload(task, businesses)
            opti_outbox.enqueue(payload)
            summary["enqueued"] += 1
        except Exception as error:
            summary["errors"] += 1
            log.warning(
                "Opti reconciliation skipped task %s after %s",
                task.get("id"),
                type(error).__name__,
            )
    return summary


async def finalize_completed_task(task_id: int) -> str:
    """Enqueue the persisted final set once, then make one best-effort delivery."""
    task_row = db.get_task(task_id)
    if task_row is None:
        return "Opti sync pending — /sync"
    task = dict(task_row)
    if (
        task.get("status") != "done"
        or task.get("opti_sync_contract_version") != SCHEMA_VERSION
    ):
        return ""
    businesses = db.get_businesses_for_bridge(task_id)
    if not businesses:
        return ""
    try:
        payload = build_payload(task, businesses)
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

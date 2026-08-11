"""Restart-safe SQLite outbox and bounded background delivery worker."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

import config
import db
from integrations.opti_client import DeliveryError, OptiClient
from integrations.opti_contract import (
    SCHEMA_VERSION,
    idempotency_key,
    payload_hash,
    serialize_payload,
)

log = logging.getLogger("lead_hunter.opti_outbox")
_MAX_BACKOFF_SECONDS = 3600


class OutboxConflict(ValueError):
    """An external batch ID was already enqueued with different immutable bytes."""


def _now(value: datetime | None = None) -> datetime:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return _now(value).isoformat().replace("+00:00", "Z")


def _redact(message: str, token: str = "") -> str:
    clean = message.replace(token, "[REDACTED]") if token else message
    clean = re.sub(r"(?i)bearer\s+[a-z0-9._~+/-]+", "Bearer [REDACTED]", clean)
    return " ".join(clean.split())[:1000]


def enqueue(payload: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    payload_bytes = serialize_payload(payload)
    digest = payload_hash(payload_bytes)
    external_batch_id = str(payload["externalBatchId"])
    key = idempotency_key(external_batch_id, digest)
    timestamp = _timestamp(now)
    with db._connect() as conn:
        existing = conn.execute(
            "SELECT * FROM opti_sync_outbox WHERE externalBatchId = ?",
            (external_batch_id,),
        ).fetchone()
        if existing is not None:
            if existing["payloadHash"] != digest or existing["payloadJson"].encode("utf-8") != payload_bytes:
                raise OutboxConflict(
                    f"external batch {external_batch_id!r} already has a different payload"
                )
            return dict(existing)
        cursor = conn.execute(
            """
            INSERT INTO opti_sync_outbox (
                externalBatchId, schemaVersion, payloadJson, payloadHash,
                idempotencyKey, status, attempts, nextAttemptAt, createdAt, updatedAt
            ) VALUES (?, ?, ?, ?, ?, 'PENDING', 0, ?, ?, ?)
            """,
            (
                external_batch_id,
                SCHEMA_VERSION,
                payload_bytes.decode("utf-8"),
                digest,
                key,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM opti_sync_outbox WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)


def get_by_batch(external_batch_id: str) -> dict[str, Any] | None:
    with db._connect() as conn:
        row = conn.execute(
            "SELECT * FROM opti_sync_outbox WHERE externalBatchId = ?",
            (external_batch_id,),
        ).fetchone()
        return dict(row) if row else None


def summary_counts() -> dict[str, int]:
    result = {status: 0 for status in ("PENDING", "RETRY", "FAILED", "SENT", "SENDING")}
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS total FROM opti_sync_outbox GROUP BY status"
        ).fetchall()
    for row in rows:
        result[row["status"]] = row["total"]
    return result


def format_summary(counts: Mapping[str, int] | None = None) -> str:
    values = counts or summary_counts()
    return (
        "Opti sync: "
        f"pending {values.get('PENDING', 0)}, retry {values.get('RETRY', 0)}, "
        f"failed {values.get('FAILED', 0)}, sent {values.get('SENT', 0)}"
    )


def recover_stale_sending(
    *, now: datetime | None = None, stale_after_seconds: int = 300
) -> int:
    current = _now(now)
    cutoff = _timestamp(current - timedelta(seconds=stale_after_seconds))
    timestamp = _timestamp(current)
    with db._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE opti_sync_outbox
            SET status = 'RETRY', nextAttemptAt = ?, updatedAt = ?,
                lastErrorCode = 'STALE_SENDING_RECOVERED',
                lastErrorMessage = 'Recovered interrupted delivery'
            WHERE status = 'SENDING' AND updatedAt <= ?
            """,
            (timestamp, timestamp, cutoff),
        )
        return cursor.rowcount


def retry_failed(*, now: datetime | None = None) -> int:
    timestamp = _timestamp(now)
    with db._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE opti_sync_outbox
            SET status = 'RETRY', attempts = 0, nextAttemptAt = ?, updatedAt = ?,
                lastErrorCode = NULL, lastErrorMessage = NULL
            WHERE status = 'FAILED'
            """,
            (timestamp, timestamp),
        )
        return cursor.rowcount


def retry_failed_batch(
    external_batch_id: str, *, now: datetime | None = None
) -> bool:
    """Reset exactly one failed batch without changing global /sync behavior."""
    timestamp = _timestamp(now)
    with db._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE opti_sync_outbox
            SET status = 'RETRY', attempts = 0, nextAttemptAt = ?, updatedAt = ?,
                lastErrorCode = NULL, lastErrorMessage = NULL
            WHERE externalBatchId = ? AND status = 'FAILED'
            """,
            (timestamp, timestamp, external_batch_id),
        )
        return cursor.rowcount == 1


def _claim_due(
    *,
    now: datetime | None = None,
    external_batch_id: str | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any] | None:
    timestamp = _timestamp(now)
    maximum = max_attempts or config.OPTI_SYNC_MAX_ATTEMPTS
    with db._connect() as conn:
        clauses = [
            "status IN ('PENDING', 'RETRY')",
            "(nextAttemptAt IS NULL OR nextAttemptAt <= ?)",
            "attempts < ?",
        ]
        parameters: list[Any] = [timestamp, maximum]
        if external_batch_id is not None:
            clauses.append("externalBatchId = ?")
            parameters.append(external_batch_id)
        row = conn.execute(
            f"""
            UPDATE opti_sync_outbox
            SET status = 'SENDING', attempts = attempts + 1, updatedAt = ?
            WHERE id = (
                SELECT id FROM opti_sync_outbox
                WHERE {' AND '.join(clauses)}
                ORDER BY id
                LIMIT 1
            )
              AND status IN ('PENDING', 'RETRY')
            RETURNING *
            """,
            [timestamp, *parameters],
        ).fetchone()
        return dict(row) if row is not None else None


def _mark_success(row_id: int, response: Mapping[str, Any], *, now: datetime | None = None) -> None:
    timestamp = _timestamp(now)
    summary_json = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(summary_json) > 100_000:
        raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response summary is too large", True)
    with db._connect() as conn:
        conn.execute(
            """
            UPDATE opti_sync_outbox
            SET status = 'SENT', nextAttemptAt = NULL, lastErrorCode = NULL,
                lastErrorMessage = NULL, optiBatchId = ?, responseSummaryJson = ?,
                updatedAt = ?, sentAt = ?
            WHERE id = ? AND status = 'SENDING'
            """,
            (response["optiBatchId"], summary_json, timestamp, timestamp, row_id),
        )


def _mark_error(
    row: Mapping[str, Any],
    error: DeliveryError,
    *,
    now: datetime | None = None,
    max_attempts: int | None = None,
    retry_base_seconds: int | None = None,
    token: str = "",
) -> None:
    current = _now(now)
    maximum = max_attempts or config.OPTI_SYNC_MAX_ATTEMPTS
    base = retry_base_seconds or config.OPTI_SYNC_RETRY_BASE_SECONDS
    attempts = int(row["attempts"])
    retryable = error.retryable and attempts < maximum
    status = "RETRY" if retryable else "FAILED"
    next_attempt = (
        _timestamp(current + timedelta(seconds=min(base * (2 ** (attempts - 1)), _MAX_BACKOFF_SECONDS)))
        if retryable
        else None
    )
    with db._connect() as conn:
        conn.execute(
            """
            UPDATE opti_sync_outbox
            SET status = ?, nextAttemptAt = ?, lastErrorCode = ?, lastErrorMessage = ?, updatedAt = ?
            WHERE id = ? AND status = 'SENDING'
            """,
            (
                status,
                next_attempt,
                error.code[:100],
                _redact(error.message, token),
                _timestamp(current),
                row["id"],
            ),
        )


async def deliver_due(
    *,
    external_batch_id: str | None = None,
    limit: int = 10,
    now: datetime | None = None,
    client_factory: Callable[[], OptiClient] | None = None,
) -> int:
    """Deliver due rows, returning successes. Disabled means exactly zero HTTP calls."""
    if not config.OPTI_BRIDGE_ENABLED:
        return 0
    successes = 0
    for _ in range(max(0, limit)):
        row = _claim_due(now=now, external_batch_id=external_batch_id)
        if row is None:
            break
        token = config.OPTI_IMPORT_TOKEN
        try:
            client = (
                client_factory()
                if client_factory
                else OptiClient(
                    config.OPTI_BASE_URL,
                    token,
                    config.OPTI_IMPORT_TIMEOUT_SECONDS,
                )
            )
            response = await client.import_batch(
                row["payloadJson"].encode("utf-8"), row["idempotencyKey"]
            )
            _mark_success(row["id"], response, now=now)
            successes += 1
        except DeliveryError as error:
            _mark_error(row, error, now=now, token=token)
        except Exception as error:  # defensive boundary: never break search/export
            _mark_error(
                row,
                DeliveryError("OPTI_CLIENT_ERROR", type(error).__name__, True),
                now=now,
                token=token,
            )
        if external_batch_id is not None:
            break
    return successes


class OutboxWorker:
    """One bounded periodic worker per application process."""

    def __init__(self, poll_seconds: float = 5.0) -> None:
        self.poll_seconds = max(1.0, poll_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        if self.running:
            return False
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="opti-outbox-worker")
        return True

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await deliver_due(limit=10)
            except Exception:
                log.exception("Opti outbox worker iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

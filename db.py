# -*- coding: utf-8 -*-
"""SQLite: хранение задач поиска и найденных бизнесов."""

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import config
from models import Business

OPTI_SYNC_CONTRACT_VERSION = "opti.lead-import.v1"
ACTIVE_TASK_STATUSES = ("new", "collecting", "checking", "scoring")
TERMINAL_TASK_STATUSES = ("done", "error", "stopped")
_MAX_PROGRESS_JSON = 10_000
_MAX_ERROR_MESSAGE = 1000


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Открыть соединение с базой (одно на операцию — безопасно для потоков)."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Создать таблицы, если их ещё нет."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                niche       TEXT NOT NULL,
                city        TEXT NOT NULL,
                count       INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'new',  -- new/collecting/checking/scoring/done/error/stopped
                chat_id     INTEGER,                       -- чат Telegram, запустивший задачу
                csv_path    TEXT,                          -- путь к готовому CSV
                created_at  TEXT NOT NULL,
                finished_at TEXT,
                external_batch_id TEXT,
                opti_sync_contract_version TEXT,
                initiated_via TEXT NOT NULL DEFAULT 'TELEGRAM',
                control_idempotency_key TEXT,
                progress_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT,
                last_error_code TEXT,
                last_error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS businesses (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id          INTEGER REFERENCES tasks(id),
                name             TEXT,
                niche            TEXT,
                city             TEXT,
                phone            TEXT,
                address          TEXT,
                category         TEXT DEFAULT '',
                email            TEXT DEFAULT '',
                website          TEXT,
                instagram_url    TEXT,
                google_maps_url  TEXT DEFAULT '',
                google_place_id  TEXT DEFAULT '',
                external_candidate_id TEXT DEFAULT '',
                collected_at     TEXT DEFAULT '',
                rating           REAL DEFAULT 0,
                reviews_count    INTEGER DEFAULT 0,
                has_site         INTEGER DEFAULT 0,
                site_quality     TEXT DEFAULT 'none',
                instagram_active INTEGER DEFAULT 0,
                followers        INTEGER DEFAULT 0,
                posts_count      INTEGER DEFAULT 0,
                last_post_days   INTEGER,
                ai_score         INTEGER DEFAULT 0,
                ai_priority      TEXT DEFAULT '',
                ai_reason        TEXT DEFAULT '',
                website_original_url TEXT DEFAULT '',
                instagram_bio_url TEXT DEFAULT '',
                website_resolved_url TEXT DEFAULT '',
                website_final_url TEXT DEFAULT '',
                website_resolution_status TEXT DEFAULT '',
                website_resolution_source TEXT DEFAULT '',
                website_resolution_confidence REAL DEFAULT 0,
                website_resolution_evidence TEXT DEFAULT '',
                website_resolution_error TEXT DEFAULT '',
                website_audit_status TEXT DEFAULT '',
                website_audit_http_status INTEGER,
                website_audit_evidence TEXT DEFAULT '',
                website_audit_error TEXT DEFAULT '',
                lead_decision TEXT DEFAULT '',
                lead_decision_reason TEXT DEFAULT '',
                website_presence_status TEXT DEFAULT '',
                website_presence_source TEXT DEFAULT '',
                website_presence_resolved_url TEXT DEFAULT '',
                website_presence_evidence TEXT DEFAULT '',
                website_presence_error TEXT DEFAULT ''
            );

            -- Дубликаты по телефону внутри одной задачи не сохраняем
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_phone
                ON businesses(task_id, phone) WHERE phone != '';

            CREATE TABLE IF NOT EXISTS opti_sync_outbox (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                externalBatchId     TEXT NOT NULL UNIQUE,
                schemaVersion       TEXT NOT NULL,
                payloadJson         TEXT NOT NULL,
                payloadHash         TEXT NOT NULL,
                idempotencyKey      TEXT NOT NULL,
                status              TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING', 'SENDING', 'SENT', 'RETRY', 'FAILED')),
                attempts            INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                nextAttemptAt       TEXT,
                lastErrorCode       TEXT,
                lastErrorMessage    TEXT,
                optiBatchId         TEXT,
                responseSummaryJson TEXT,
                createdAt           TEXT NOT NULL,
                updatedAt           TEXT NOT NULL,
                sentAt              TEXT,
                CHECK (
                    length(payloadHash) = 64
                    AND payloadHash = lower(payloadHash)
                    AND payloadHash NOT GLOB '*[^0-9a-f]*'
                ),
                CHECK (length(lastErrorMessage) <= 1000)
            );

            CREATE INDEX IF NOT EXISTS idx_opti_outbox_due
                ON opti_sync_outbox(status, nextAttemptAt, id);

            CREATE TRIGGER IF NOT EXISTS trg_opti_outbox_payload_immutable
            BEFORE UPDATE OF externalBatchId, schemaVersion, payloadJson,
                payloadHash, idempotencyKey ON opti_sync_outbox
            BEGIN
                SELECT RAISE(ABORT, 'Opti outbox payload is immutable');
            END;
            """
        )
        _migrate_tasks(conn)
        _migrate_businesses(conn)
        from candidate_history import (
            backfill_persisted_leads,
            initialize_history_schema,
            reconcile_city_alias_scopes,
        )

        initialize_history_schema(conn)
        reconcile_city_alias_scopes(conn)
        backfill_persisted_leads(conn)


_BUSINESS_MIGRATIONS = {
    "category": "TEXT DEFAULT ''",
    "email": "TEXT DEFAULT ''",
    "google_maps_url": "TEXT DEFAULT ''",
    "google_place_id": "TEXT DEFAULT ''",
    "external_candidate_id": "TEXT DEFAULT ''",
    "collected_at": "TEXT DEFAULT ''",
    "website_original_url": "TEXT DEFAULT ''",
    "instagram_bio_url": "TEXT DEFAULT ''",
    "website_resolved_url": "TEXT DEFAULT ''",
    "website_final_url": "TEXT DEFAULT ''",
    "website_resolution_status": "TEXT DEFAULT ''",
    "website_resolution_source": "TEXT DEFAULT ''",
    "website_resolution_confidence": "REAL DEFAULT 0",
    "website_resolution_evidence": "TEXT DEFAULT ''",
    "website_resolution_error": "TEXT DEFAULT ''",
    "website_audit_status": "TEXT DEFAULT ''",
    "website_audit_http_status": "INTEGER",
    "website_audit_evidence": "TEXT DEFAULT ''",
    "website_audit_error": "TEXT DEFAULT ''",
    "lead_decision": "TEXT DEFAULT ''",
    "lead_decision_reason": "TEXT DEFAULT ''",
    "website_presence_status": "TEXT DEFAULT ''",
    "website_presence_source": "TEXT DEFAULT ''",
    "website_presence_resolved_url": "TEXT DEFAULT ''",
    "website_presence_evidence": "TEXT DEFAULT ''",
    "website_presence_error": "TEXT DEFAULT ''",
}


def _migrate_tasks(conn: sqlite3.Connection) -> None:
    """Add a durable external batch identity without changing local task IDs."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "external_batch_id" not in existing:
        conn.execute("ALTER TABLE tasks ADD COLUMN external_batch_id TEXT")
    if "opti_sync_contract_version" not in existing:
        conn.execute("ALTER TABLE tasks ADD COLUMN opti_sync_contract_version TEXT")
    additions = {
        "initiated_via": "TEXT NOT NULL DEFAULT 'TELEGRAM'",
        "control_idempotency_key": "TEXT",
        "progress_json": "TEXT NOT NULL DEFAULT '{}'",
        "updated_at": "TEXT",
        "last_error_code": "TEXT",
        "last_error_message": "TEXT",
    }
    for name, declaration in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {declaration}")
    conn.execute(
        "UPDATE tasks SET initiated_via = 'TELEGRAM' "
        "WHERE initiated_via IS NULL OR initiated_via NOT IN ('TELEGRAM', 'OPTI')"
    )
    conn.execute(
        "UPDATE tasks SET progress_json = '{}' WHERE progress_json IS NULL"
    )
    conn.execute(
        "UPDATE tasks SET updated_at = created_at WHERE updated_at IS NULL"
    )
    rows = conn.execute(
        "SELECT id, created_at FROM tasks WHERE external_batch_id IS NULL"
    ).fetchall()
    for row in rows:
        seed = f"legacy:{row['id']}:{row['created_at']}".encode("utf-8")
        suffix = hashlib.sha256(seed).hexdigest()[:24]
        conn.execute(
            "UPDATE tasks SET external_batch_id = ? WHERE id = ?",
            (f"legacy-task-{suffix}", row["id"]),
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_external_batch_id "
        "ON tasks(external_batch_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_control_idempotency_key "
        "ON tasks(control_idempotency_key) WHERE control_idempotency_key IS NOT NULL"
    )


def _migrate_businesses(conn: sqlite3.Connection) -> None:
    """Add Phase 3 columns to an existing table without rebuilding it."""
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(businesses)")
    }
    for name, declaration in _BUSINESS_MIGRATIONS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE businesses ADD COLUMN {name} {declaration}")


# ---------- Задачи ----------

def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _safe_error_message(message: object) -> str:
    clean = re.sub(r"(?i)bearer\s+[^\s]+", "Bearer [REDACTED]", str(message))
    return " ".join(clean.split())[:_MAX_ERROR_MESSAGE]


def create_task(
    niche: str,
    city: str,
    count: int,
    chat_id: Optional[int] = None,
    *,
    initiated_via: str = "TELEGRAM",
    control_idempotency_key: str | None = None,
) -> int:
    """Создать задачу поиска, вернуть её id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (niche, city, count, status, chat_id, created_at, "
            "external_batch_id, opti_sync_contract_version, initiated_via, "
            "control_idempotency_key, progress_json, updated_at) "
            "VALUES (?, ?, ?, 'new', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                niche,
                city,
                count,
                chat_id,
                _timestamp(),
                str(uuid.uuid4()),
                OPTI_SYNC_CONTRACT_VERSION,
                initiated_via,
                control_idempotency_key,
                json.dumps({"stage": "new", "targetLeads": count}),
                _timestamp(),
            ),
        )
        return cur.lastrowid


def update_task_status(
    task_id: int,
    status: str,
    csv_path: Optional[str] = None,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """Обновить статус задачи; для финальных статусов проставить время окончания."""
    finished = (
        _timestamp()
        if status in TERMINAL_TASK_STATUSES
        else None
    )
    with _connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assignments = [
            "status = ?",
            "csv_path = COALESCE(?, csv_path)",
            "finished_at = COALESCE(?, finished_at)",
        ]
        parameters: list[object] = [status, csv_path, finished]
        if "updated_at" in columns:
            assignments.append("updated_at = ?")
            parameters.append(_timestamp())
        if "last_error_code" in columns:
            assignments.append("last_error_code = ?")
            parameters.append(error_code[:100] if error_code else None)
        if "last_error_message" in columns:
            assignments.append("last_error_message = ?")
            parameters.append(_safe_error_message(error_message) if error_message else None)
        parameters.append(task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(assignments)} WHERE id = ?",
            parameters,
        )


def update_task_progress(task_id: int, progress: dict) -> None:
    payload = json.dumps(
        progress,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(payload.encode("utf-8")) > _MAX_PROGRESS_JSON:
        raise ValueError("progress_json exceeds the bounded size")
    with _connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        if "progress_json" not in columns:
            return
        updated = ", updated_at = ?" if "updated_at" in columns else ""
        parameters: tuple[object, ...] = (
            (payload, _timestamp(), task_id)
            if updated
            else (payload, task_id)
        )
        conn.execute(
            f"UPDATE tasks SET progress_json = ?{updated} WHERE id = ?",
            parameters,
        )


def get_task(task_id: int) -> Optional[sqlite3.Row]:
    """Получить задачу по id."""
    with _connect() as conn:
        return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def get_task_by_search_id(search_id: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE external_batch_id = ?", (search_id,)
        ).fetchone()


def get_task_by_control_key(key: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE control_idempotency_key = ?", (key,)
        ).fetchone()


def list_tasks(limit: int = 50) -> List[sqlite3.Row]:
    bounded = max(1, min(int(limit), 100))
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (bounded,)
        ).fetchall()


def recover_interrupted_tasks() -> int:
    timestamp = _timestamp()
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE tasks
            SET status = 'error', finished_at = ?, updated_at = ?,
                last_error_code = 'PROCESS_RESTARTED',
                last_error_message = 'Search was interrupted by service restart'
            WHERE status IN ('new', 'collecting', 'checking', 'scoring')
            """,
            (timestamp, timestamp),
        )
        return cursor.rowcount


def get_last_task(chat_id: Optional[int] = None) -> Optional[sqlite3.Row]:
    """Последняя задача (для /status и /export)."""
    with _connect() as conn:
        if chat_id is not None:
            return conn.execute(
                "SELECT * FROM tasks WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        return conn.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT 1").fetchone()


def get_completed_tasks_for_opti_reconciliation(limit: int = 100) -> List[sqlite3.Row]:
    """Return eligible completed tasks that do not yet have a durable outbox row."""
    bounded_limit = max(0, min(int(limit), 1000))
    with _connect() as conn:
        return conn.execute(
            """
            SELECT tasks.*
            FROM tasks
            WHERE tasks.status = 'done'
              AND tasks.opti_sync_contract_version = ?
              AND EXISTS (
                  SELECT 1 FROM businesses WHERE businesses.task_id = tasks.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM opti_sync_outbox
                  WHERE opti_sync_outbox.externalBatchId = tasks.external_batch_id
              )
            ORDER BY tasks.id ASC
            LIMIT ?
            """,
            (OPTI_SYNC_CONTRACT_VERSION, bounded_limit),
        ).fetchall()


# ---------- Бизнесы ----------

_BUSINESS_COLUMNS = (
    "task_id", "name", "niche", "city", "phone", "address", "category", "email", "website",
    "instagram_url", "google_maps_url", "google_place_id", "external_candidate_id",
    "collected_at", "rating", "reviews_count", "has_site", "site_quality",
    "instagram_active", "followers", "posts_count", "last_post_days",
    "ai_score", "ai_priority", "ai_reason",
    "website_original_url", "instagram_bio_url", "website_resolved_url",
    "website_final_url", "website_resolution_status", "website_resolution_source",
    "website_resolution_confidence", "website_resolution_evidence",
    "website_resolution_error", "website_audit_status",
    "website_audit_http_status", "website_audit_evidence", "website_audit_error",
    "lead_decision", "lead_decision_reason",
    "website_presence_status", "website_presence_source",
    "website_presence_resolved_url", "website_presence_evidence",
    "website_presence_error",
)


def save_businesses(businesses: List[Business]) -> int:
    """Сохранить список бизнесов. Дубликаты по телефону пропускаются. Вернуть число сохранённых."""
    saved = 0
    with _connect() as conn:
        for b in businesses:
            data = b.to_dict()
            values = tuple(data[col] for col in _BUSINESS_COLUMNS)
            placeholders = ", ".join("?" for _ in _BUSINESS_COLUMNS)
            try:
                cur = conn.execute(
                    f"INSERT INTO businesses ({', '.join(_BUSINESS_COLUMNS)}) "
                    f"VALUES ({placeholders})",
                    values,
                )
                b.id = cur.lastrowid
                saved += 1
            except sqlite3.IntegrityError:
                # Дубликат телефона внутри задачи — пропускаем
                continue
    return saved


def update_business(b: Business) -> None:
    """Обновить запись бизнеса после проверок/скоринга."""
    if b.id is None:
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE businesses SET website=?, has_site=?, site_quality=?, instagram_active=?, "
            "followers=?, posts_count=?, last_post_days=?, ai_score=?, ai_priority=?, ai_reason=?, "
            "instagram_url=?, website_original_url=?, instagram_bio_url=?, "
            "website_resolved_url=?, website_final_url=?, website_resolution_status=?, "
            "website_resolution_source=?, website_resolution_confidence=?, "
            "website_resolution_evidence=?, website_resolution_error=?, website_audit_status=?, "
            "website_audit_http_status=?, website_audit_evidence=?, website_audit_error=?, "
            "lead_decision=?, lead_decision_reason=?, website_presence_status=?, "
            "website_presence_source=?, website_presence_resolved_url=?, "
            "website_presence_evidence=?, website_presence_error=? WHERE id=?",
            (
                b.website, int(b.has_site), b.site_quality, int(b.instagram_active),
                b.followers, b.posts_count, b.last_post_days,
                b.ai_score, b.ai_priority, b.ai_reason,
                b.instagram_url, b.website_original_url, b.instagram_bio_url,
                b.website_resolved_url, b.website_final_url,
                b.website_resolution_status, b.website_resolution_source,
                b.website_resolution_confidence, b.website_resolution_evidence,
                b.website_resolution_error, b.website_audit_status,
                b.website_audit_http_status, b.website_audit_evidence,
                b.website_audit_error, b.lead_decision, b.lead_decision_reason,
                b.website_presence_status, b.website_presence_source,
                b.website_presence_resolved_url, b.website_presence_evidence,
                b.website_presence_error, b.id,
            ),
        )


def get_businesses(task_id: int) -> List[Business]:
    """Все бизнесы задачи в виде объектов Business."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM businesses WHERE task_id = ? ORDER BY ai_score DESC",
            (task_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["has_site"] = bool(d["has_site"])
        d["instagram_active"] = bool(d["instagram_active"])
        result.append(Business(**d))
    return result


def get_businesses_for_bridge(task_id: int) -> List[Business]:
    """Return the persisted final set in stable insertion/rank order."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM businesses WHERE task_id = ? ORDER BY id ASC",
            (task_id,),
        ).fetchall()
    result = []
    for row in rows:
        data = dict(row)
        data["has_site"] = bool(data["has_site"])
        data["instagram_active"] = bool(data["instagram_active"])
        result.append(Business(**data))
    return result

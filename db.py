# -*- coding: utf-8 -*-
"""SQLite: хранение задач поиска и найденных бизнесов."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional

import config
from models import Business


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
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS businesses (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id          INTEGER REFERENCES tasks(id),
                name             TEXT,
                niche            TEXT,
                city             TEXT,
                phone            TEXT,
                address          TEXT,
                website          TEXT,
                instagram_url    TEXT,
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
                lead_decision_reason TEXT DEFAULT ''
            );

            -- Дубликаты по телефону внутри одной задачи не сохраняем
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_phone
                ON businesses(task_id, phone) WHERE phone != '';
            """
        )
        _migrate_businesses(conn)


_BUSINESS_MIGRATIONS = {
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
}


def _migrate_businesses(conn: sqlite3.Connection) -> None:
    """Add Phase 3 columns to an existing table without rebuilding it."""
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(businesses)")
    }
    for name, declaration in _BUSINESS_MIGRATIONS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE businesses ADD COLUMN {name} {declaration}")


# ---------- Задачи ----------

def create_task(niche: str, city: str, count: int, chat_id: Optional[int] = None) -> int:
    """Создать задачу поиска, вернуть её id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (niche, city, count, status, chat_id, created_at) "
            "VALUES (?, ?, ?, 'new', ?, ?)",
            (niche, city, count, chat_id, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def update_task_status(task_id: int, status: str, csv_path: Optional[str] = None) -> None:
    """Обновить статус задачи; для финальных статусов проставить время окончания."""
    finished = (
        datetime.now().isoformat(timespec="seconds")
        if status in ("done", "error", "stopped")
        else None
    )
    with _connect() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, "
            "csv_path = COALESCE(?, csv_path), "
            "finished_at = COALESCE(?, finished_at) WHERE id = ?",
            (status, csv_path, finished, task_id),
        )


def get_task(task_id: int) -> Optional[sqlite3.Row]:
    """Получить задачу по id."""
    with _connect() as conn:
        return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def get_last_task(chat_id: Optional[int] = None) -> Optional[sqlite3.Row]:
    """Последняя задача (для /status и /export)."""
    with _connect() as conn:
        if chat_id is not None:
            return conn.execute(
                "SELECT * FROM tasks WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        return conn.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT 1").fetchone()


# ---------- Бизнесы ----------

_BUSINESS_COLUMNS = (
    "task_id", "name", "niche", "city", "phone", "address", "website",
    "instagram_url", "rating", "reviews_count", "has_site", "site_quality",
    "instagram_active", "followers", "posts_count", "last_post_days",
    "ai_score", "ai_priority", "ai_reason",
    "website_original_url", "instagram_bio_url", "website_resolved_url",
    "website_final_url", "website_resolution_status", "website_resolution_source",
    "website_resolution_confidence", "website_resolution_evidence",
    "website_resolution_error", "website_audit_status",
    "website_audit_http_status", "website_audit_evidence", "website_audit_error",
    "lead_decision", "lead_decision_reason",
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
            "lead_decision=?, lead_decision_reason=? WHERE id=?",
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
                b.website_audit_error, b.lead_decision, b.lead_decision_reason, b.id,
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

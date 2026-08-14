"""Durable, privacy-safe candidate history scoped by canonical niche and city."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import sqlite3
import unicodedata
from typing import Any, Mapping

import city_catalog
import config
from city_catalog import CityDefinition
from contactability import normalize_phone, normalized_instagram_profile
from niche_catalog import NicheSearchPlan, resolve_niche_plan
from website_resolution import normalize_domain


class CandidateClaimResult(str, Enum):
    CLAIMED = "claimed"
    ALREADY_CHECKED = "already_checked"
    CLAIMED_BY_OTHER_TASK = "claimed_by_other_task"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _normalized_text(value: object) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(raw.split())


def canonical_scope_key(niche: str, city: str) -> str:
    niche_plan = resolve_niche_plan(niche)
    city_definition = city_catalog.resolve_city(city, city_catalog.CITY_DEFINITIONS)
    return canonical_scope_key_from_resolved(niche_plan, city_definition, city)


def canonical_scope_key_from_resolved(
    niche_plan: NicheSearchPlan,
    city_definition: CityDefinition | None,
    city_input: str,
) -> str:
    niche_identity = (
        str(niche_plan.key)
        if niche_plan.known
        else _normalized_text(niche_plan.base_niche)
    )
    city_identity = (
        city_definition.key
        if city_definition is not None
        else _normalized_text(city_input)
    )
    digest = hashlib.sha256(f"v1\x1f{niche_identity}\x1f{city_identity}".encode()).hexdigest()
    return f"scope:v1:{digest}"


def _get(candidate: object, name: str) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, "")
    return getattr(candidate, name, "")


def _maps_url(value: object) -> str:
    from integrations.opti_contract import normalize_google_maps_url

    return normalize_google_maps_url(value)


def _instagram_username(value: object) -> str:
    normalized = normalized_instagram_profile(value)
    return normalized.rstrip("/").rsplit("/", 1)[-1] if normalized else ""


def candidate_fingerprint(candidate: object, canonical_city: str) -> tuple[str, str]:
    choices: tuple[tuple[str, str], ...] = (
        ("place_id", _normalized_text(_get(candidate, "google_place_id"))),
        ("maps_url", _maps_url(_get(candidate, "google_maps_url"))),
        ("phone", normalize_phone(_get(candidate, "phone")) or ""),
        ("instagram", _instagram_username(_get(candidate, "instagram_url"))),
    )
    basis = ""
    identity = ""
    for candidate_basis, value in choices:
        if value:
            basis, identity = candidate_basis, value
            break
    if not identity:
        website = _get(candidate, "website")
        try:
            domain = normalize_domain(website) if website else ""
        except (TypeError, ValueError):
            domain = ""
        if domain:
            basis, identity = "website_domain", domain
    if not identity:
        basis = "name_city_address"
        identity = "\x1f".join(
            (
                _normalized_text(_get(candidate, "name")),
                _normalized_text(canonical_city),
                _normalized_text(_get(candidate, "address")),
            )
        )
    digest = hashlib.sha256(f"v1\x1f{basis}\x1f{identity}".encode()).hexdigest()
    return basis, f"cand:v1:{basis}:{digest}"


def initialize_history_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS candidate_history (
            scope_key TEXT NOT NULL,
            candidate_key TEXT NOT NULL,
            identity_basis TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('claimed','checked')),
            claimed_by_task_id INTEGER,
            claim_expires_at TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            checked_at TEXT,
            last_outcome TEXT,
            times_seen INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(scope_key,candidate_key)
        );
        CREATE INDEX IF NOT EXISTS idx_candidate_history_state_expiry
            ON candidate_history(state, claim_expires_at);
        CREATE INDEX IF NOT EXISTS idx_candidate_history_task
            ON candidate_history(claimed_by_task_id);
        CREATE INDEX IF NOT EXISTS idx_candidate_history_scope_state
            ON candidate_history(scope_key, state);
        """
    )


def backfill_persisted_leads(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT businesses.*, tasks.niche AS task_niche, tasks.city AS task_city "
        "FROM businesses JOIN tasks ON tasks.id = businesses.task_id"
    ).fetchall()
    inserted = 0
    timestamp = _iso(_now())
    for row in rows:
        data = dict(row)
        niche_plan = resolve_niche_plan(data["task_niche"])
        city_definition = city_catalog.resolve_city(
            data["task_city"], city_catalog.CITY_DEFINITIONS
        )
        scope = canonical_scope_key_from_resolved(
            niche_plan, city_definition, data["task_city"]
        )
        canonical_city = (
            city_definition.canonical_name
            if city_definition is not None
            else data["task_city"]
        )
        basis, key = candidate_fingerprint(data, canonical_city)
        cursor = conn.execute(
            "INSERT OR IGNORE INTO candidate_history "
            "(scope_key,candidate_key,identity_basis,state,claimed_by_task_id,claim_expires_at,"
            "first_seen_at,last_seen_at,checked_at,last_outcome,times_seen) "
            "VALUES (?,?,?,'checked',NULL,NULL,?,?,?,?,1)",
            (scope, key, basis, timestamp, timestamp, timestamp, "legacy_persisted_lead"),
        )
        inserted += cursor.rowcount
    return inserted


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def claim_candidate(
    scope_key: str,
    candidate_key: str,
    identity_basis: str,
    task_id: int,
    *,
    lease_seconds: int | None = None,
    now: datetime | None = None,
) -> CandidateClaimResult:
    current = now or _now()
    lease = lease_seconds if lease_seconds is not None else config.CANDIDATE_HISTORY_CLAIM_LEASE_SECONDS
    timestamp = _iso(current)
    expires = _iso(current + timedelta(seconds=lease))
    conn = _connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM candidate_history WHERE scope_key=? AND candidate_key=?",
            (scope_key, candidate_key),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO candidate_history "
                "(scope_key,candidate_key,identity_basis,state,claimed_by_task_id,claim_expires_at,"
                "first_seen_at,last_seen_at,times_seen) VALUES (?,?,?,'claimed',?,?,?,?,1)",
                (scope_key, candidate_key, identity_basis, task_id, expires, timestamp, timestamp),
            )
            result = CandidateClaimResult.CLAIMED
        elif row["state"] == "checked":
            conn.execute(
                "UPDATE candidate_history SET last_seen_at=?, times_seen=times_seen+1 "
                "WHERE scope_key=? AND candidate_key=?",
                (timestamp, scope_key, candidate_key),
            )
            result = CandidateClaimResult.ALREADY_CHECKED
        elif row["claimed_by_task_id"] == task_id or (
            row["claim_expires_at"] and _parse(row["claim_expires_at"]) <= current
        ):
            conn.execute(
                "UPDATE candidate_history SET identity_basis=?, claimed_by_task_id=?, "
                "claim_expires_at=?, last_seen_at=?, times_seen=times_seen+1 "
                "WHERE scope_key=? AND candidate_key=?",
                (identity_basis, task_id, expires, timestamp, scope_key, candidate_key),
            )
            result = CandidateClaimResult.CLAIMED
        else:
            conn.execute(
                "UPDATE candidate_history SET last_seen_at=?, times_seen=times_seen+1 "
                "WHERE scope_key=? AND candidate_key=?",
                (timestamp, scope_key, candidate_key),
            )
            result = CandidateClaimResult.CLAIMED_BY_OTHER_TASK
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_candidate_checked(
    scope_key: str,
    candidate_key: str,
    task_id: int,
    outcome: str,
) -> bool:
    timestamp = _iso(_now())
    conn = _connection()
    try:
        cursor = conn.execute(
            "UPDATE candidate_history SET state='checked', claimed_by_task_id=NULL, "
            "claim_expires_at=NULL, checked_at=?, last_seen_at=?, last_outcome=? "
            "WHERE scope_key=? AND candidate_key=? AND state='claimed' AND claimed_by_task_id=?",
            (timestamp, timestamp, outcome, scope_key, candidate_key, task_id),
        )
        return cursor.rowcount == 1
    finally:
        conn.close()


def release_candidate_claim(scope_key: str, candidate_key: str, task_id: int) -> bool:
    conn = _connection()
    try:
        cursor = conn.execute(
            "DELETE FROM candidate_history WHERE scope_key=? AND candidate_key=? "
            "AND state='claimed' AND claimed_by_task_id=?",
            (scope_key, candidate_key, task_id),
        )
        return cursor.rowcount == 1
    finally:
        conn.close()


def release_unfinished_candidate_claims(task_id: int) -> int:
    conn = _connection()
    try:
        cursor = conn.execute(
            "DELETE FROM candidate_history WHERE state='claimed' AND claimed_by_task_id=?",
            (task_id,),
        )
        return cursor.rowcount
    finally:
        conn.close()

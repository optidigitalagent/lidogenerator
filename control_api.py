"""Authenticated same-process HTTP control surface for Opti."""

from __future__ import annotations

import hmac
import json
import logging
import sqlite3
from collections.abc import Mapping
from typing import Any

from aiohttp import ContentTypeError, web

import config
import db
from integrations import opti_outbox
from search_runtime import SearchRuntime

log = logging.getLogger("lead_hunter.control_api")
PREFIX = "/internal/opti/v1"
_CREATE_FIELDS = {"niche", "city", "targetLeads"}
_TERMINAL = {"done", "error", "stopped"}


class ControlError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message}}, status=status
    )


@web.middleware
async def _error_boundary(request: web.Request, handler):
    try:
        return await handler(request)
    except ControlError as error:
        return _error(error.status, error.code, error.message)
    except web.HTTPException:
        raise
    except Exception:
        log.exception("Unhandled Opti control request failure")
        return _error(
            500, "INTERNAL_SERVER_ERROR", "The request could not be completed"
        )


@web.middleware
async def _authenticate(request: web.Request, handler):
    if request.path == "/health":
        return await handler(request)
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return _error(401, "AUTH_REQUIRED", "Bearer authorization is required")
    supplied = header[7:]
    expected = request.app[CONTROL_TOKEN_KEY]
    if not supplied or not hmac.compare_digest(supplied, expected):
        return _error(401, "INVALID_CONTROL_TOKEN", "The control token is invalid")
    return await handler(request)


CONTROL_TOKEN_KEY = web.AppKey("control_token", str)
RUNTIME_KEY = web.AppKey("runtime", SearchRuntime)


def _safe_summary(raw: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "createdCount": None,
        "updatedCount": None,
        "duplicateCount": None,
        "rejectedCount": None,
    }
    if not isinstance(raw, str) or len(raw) > 100_000:
        return result
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return result
    if not isinstance(value, Mapping):
        return result
    for field in result:
        candidate = value.get(field)
        if type(candidate) is int and 0 <= candidate <= 1_000_000:
            result[field] = candidate
    return result


def search_dto(task: Mapping[str, Any]) -> dict[str, Any]:
    search_id = str(task["external_batch_id"])
    outbox = opti_outbox.get_by_batch(search_id)
    sync_status = str(outbox["status"]) if outbox else "NOT_ENQUEUED"
    summary = _safe_summary(outbox.get("responseSummaryJson") if outbox else None)
    try:
        progress = json.loads(task.get("progress_json") or "{}")
    except (TypeError, ValueError):
        progress = {}
    if not isinstance(progress, dict):
        progress = {}
    error = None
    if task.get("last_error_code") or task.get("last_error_message"):
        error = {
            "code": task.get("last_error_code") or "SEARCH_ERROR",
            "message": task.get("last_error_message") or "Search failed",
        }
    opti_batch_id = outbox.get("optiBatchId") if outbox else None
    return {
        "searchId": search_id,
        "niche": task["niche"],
        "city": task["city"],
        "targetLeads": int(task["count"]),
        "status": task["status"],
        "initiatedVia": task.get("initiated_via") or "TELEGRAM",
        "createdAt": task["created_at"],
        "updatedAt": task.get("updated_at") or task["created_at"],
        "finishedAt": task.get("finished_at"),
        "progress": progress,
        "error": error,
        "sync": {
            "status": sync_status,
            "attempts": int(outbox.get("attempts", 0)) if outbox else 0,
            "lastErrorCode": outbox.get("lastErrorCode") if outbox else None,
            "lastErrorMessage": outbox.get("lastErrorMessage") if outbox else None,
            "optiBatchId": opti_batch_id,
            **summary,
        },
        "availableActions": {
            "canStop": task["status"] not in _TERMINAL,
            "canRetrySync": sync_status == "FAILED" and task["status"] == "done",
            "canOpenOutreach": sync_status == "SENT" and bool(opti_batch_id),
        },
    }


def _task_by_search_id(search_id: str) -> dict[str, Any]:
    row = db.get_task_by_search_id(search_id)
    if row is None:
        raise ControlError(404, "NOT_FOUND", "Search was not found")
    return dict(row)


def _normalized_create(value: Any) -> tuple[str, str, int]:
    if not isinstance(value, dict) or set(value) != _CREATE_FIELDS:
        raise ControlError(400, "VALIDATION_ERROR", "Request fields are invalid")
    niche = value.get("niche")
    city = value.get("city")
    target = value.get("targetLeads")
    if not isinstance(niche, str) or not 1 <= len(niche.strip()) <= 200:
        raise ControlError(
            400, "VALIDATION_ERROR", "niche must contain 1 to 200 characters"
        )
    if not isinstance(city, str) or not 1 <= len(city.strip()) <= 200:
        raise ControlError(
            400, "VALIDATION_ERROR", "city must contain 1 to 200 characters"
        )
    if type(target) is not int or not 1 <= target <= 200:
        raise ControlError(
            400, "VALIDATION_ERROR", "targetLeads must be an integer from 1 to 200"
        )
    return niche.strip(), city.strip(), target


async def health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "controlEnabled": True,
            "activeSearches": request.app[RUNTIME_KEY].active_count,
        }
    )


async def list_searches(request: web.Request) -> web.Response:
    raw_limit = request.query.get("limit", "50")
    try:
        limit = int(raw_limit)
    except ValueError:
        raise ControlError(
            400, "VALIDATION_ERROR", "limit must be an integer"
        ) from None
    if not 1 <= limit <= 100:
        raise ControlError(400, "VALIDATION_ERROR", "limit must be between 1 and 100")
    runtime = request.app[RUNTIME_KEY]
    return web.json_response(
        {
            "searches": [search_dto(dict(row)) for row in db.list_tasks(limit)],
            "activeCount": runtime.active_count,
            "maxConcurrentSearches": runtime.max_concurrent_searches,
        }
    )


async def get_search(request: web.Request) -> web.Response:
    return web.json_response(
        search_dto(_task_by_search_id(request.match_info["search_id"]))
    )


async def create_search(request: web.Request) -> web.Response:
    key = request.headers.get("Idempotency-Key", "").strip()
    if not 1 <= len(key) <= 200:
        raise ControlError(400, "VALIDATION_ERROR", "Idempotency-Key is required")
    try:
        body = await request.json(loads=json.loads)
    except (ContentTypeError, json.JSONDecodeError, UnicodeDecodeError):
        raise ControlError(
            400, "VALIDATION_ERROR", "Request body must be valid JSON"
        ) from None
    niche, city, target = _normalized_create(body)
    existing = db.get_task_by_control_key(key)
    if existing is not None:
        row = dict(existing)
        if (row["niche"], row["city"], int(row["count"])) != (niche, city, target):
            raise ControlError(
                409,
                "IDEMPOTENCY_KEY_REUSED",
                "Idempotency-Key was used with a different request",
            )
        return web.json_response(search_dto(row), status=200)
    try:
        task_id = db.create_task(
            niche,
            city,
            target,
            initiated_via="OPTI",
            control_idempotency_key=key,
        )
    except sqlite3.IntegrityError:
        existing = db.get_task_by_control_key(key)
        if existing is None:
            raise
        row = dict(existing)
        if (row["niche"], row["city"], int(row["count"])) != (niche, city, target):
            raise ControlError(
                409,
                "IDEMPOTENCY_KEY_REUSED",
                "Idempotency-Key was used with a different request",
            )
        return web.json_response(search_dto(row), status=200)
    if not request.app[RUNTIME_KEY].launch(task_id):
        db.update_task_status(
            task_id,
            "error",
            error_code="SEARCH_STATE_CONFLICT",
            error_message="Search could not be launched",
        )
        raise ControlError(409, "SEARCH_STATE_CONFLICT", "Search could not be launched")
    return web.json_response(search_dto(dict(db.get_task(task_id))), status=201)


async def stop_search(request: web.Request) -> web.Response:
    task = _task_by_search_id(request.match_info["search_id"])
    if task["status"] not in _TERMINAL:
        request.app[RUNTIME_KEY].stop(task["id"])
    return web.json_response(search_dto(dict(db.get_task(task["id"]))))


async def retry_sync(request: web.Request) -> web.Response:
    task = _task_by_search_id(request.match_info["search_id"])
    outbox = opti_outbox.get_by_batch(task["external_batch_id"])
    if task["status"] != "done" or outbox is None:
        raise ControlError(
            409, "SEARCH_STATE_CONFLICT", "This search is not eligible for sync retry"
        )
    if outbox["status"] == "FAILED":
        opti_outbox.retry_failed_batch(task["external_batch_id"])
        await opti_outbox.deliver_due(
            external_batch_id=task["external_batch_id"], limit=1
        )
    return web.json_response(search_dto(task))


def create_app(runtime: SearchRuntime, token: str) -> web.Application:
    if len(token) < 32:
        raise ValueError("control token must contain at least 32 characters")
    app = web.Application(
        middlewares=[_error_boundary, _authenticate],
        client_max_size=16 * 1024,
    )
    app[CONTROL_TOKEN_KEY] = token
    app[RUNTIME_KEY] = runtime
    app.add_routes(
        [
            web.get("/health", health),
            web.get(f"{PREFIX}/searches", list_searches),
            web.post(f"{PREFIX}/searches", create_search),
            web.get(f"{PREFIX}/searches/{{search_id}}", get_search),
            web.post(f"{PREFIX}/searches/{{search_id}}/stop", stop_search),
            web.post(f"{PREFIX}/searches/{{search_id}}/retry-sync", retry_sync),
        ]
    )
    return app


class ControlServer:
    def __init__(self, runtime: SearchRuntime) -> None:
        self.runtime = runtime
        self._runner: web.AppRunner | None = None

    async def start(self) -> bool:
        if not config.LEAD_GENERATOR_CONTROL_ENABLED:
            return False
        app = create_app(self.runtime, config.LEAD_GENERATOR_CONTROL_TOKEN)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, config.HOST, config.PORT)
        await site.start()
        log.info("Opti control API listening on %s:%s", config.HOST, config.PORT)
        return True

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

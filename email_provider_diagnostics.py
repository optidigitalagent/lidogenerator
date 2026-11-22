"""Allowlisted server-only diagnostics; never changes provider error handling."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC
from email.utils import format_datetime, parsedate_to_datetime
from uuid import UUID

log = logging.getLogger("lead_hunter.email_source_adapter")

_CODE_CLASSES = {
    "credit_balance_exhausted": "CREDIT_BALANCE_EXHAUSTED",
    "organization_spend_limit_exceeded": "ORGANIZATION_SPEND_LIMIT_EXCEEDED",
    "project_spend_limit_exceeded": "PROJECT_SPEND_LIMIT_EXCEEDED",
    "organization_usage_limit_exceeded": "ORGANIZATION_USAGE_LIMIT_EXCEEDED",
    "rate_limit_exceeded": "RATE_LIMIT_EXCEEDED",
    "slow_down": "RATE_LIMIT_EXCEEDED",
    "rate_limit_error": "RATE_LIMIT_EXCEEDED",
    "insufficient_quota": "INSUFFICIENT_QUOTA",
}
_COUNTERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "retry-after-ms",
)
_RESETS = ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")


def _atom(value):
    # Preserve unknown safe codes, not arbitrary provider text or SDK str(dict).
    if (
        isinstance(value, str)
        and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", value)
        and not value.startswith(("sk_", "sess_", "token_"))
    ):
        return value
    return None


def _uuid(value):
    if isinstance(value, str) and len(value) == 36:
        try:
            return str(UUID(value))
        except ValueError:
            pass
    return None


def _headers(response):
    source = getattr(response, "headers", {})
    result = {}
    for name in _COUNTERS:
        value = source.get(name)
        if isinstance(value, str) and re.fullmatch(r"[0-9]{1,15}", value):
            result[name] = value
    for name in _RESETS:
        value = source.get(name)
        if (
            isinstance(value, str)
            and len(value) <= 64
            and re.fullmatch(r"(?:[0-9]{1,9}(?:\.[0-9]{1,3})?(?:ms|s|m|h|d))+", value)
        ):
            result[name] = value
    value = source.get("retry-after")
    if isinstance(value, str) and len(value) <= 100 and value.isascii():
        if re.fullmatch(r"[0-9]{1,15}", value):
            result["retry-after"] = value
        elif not any(c in value for c in "\r\n\x00"):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is not None:
                    result["retry-after"] = format_datetime(
                        date.astimezone(UTC), usegmt=True
                    )
            except (ValueError, TypeError, OverflowError):
                pass
    return result


def log_provider_error(error, operation_key):
    """At most one bounded JSON event per original SDK failure, best effort.

    Code is authoritative for classification; type/header hints never promote an
    absent or unfamiliar code to a billing diagnosis. No body/message, query,
    credential, tenant identity, URL or client-visible details are retained.
    """
    try:
        body = getattr(error, "body", None)
        code = _atom(body.get("code")) if isinstance(body, dict) else None
        kind = _atom(body.get("type")) if isinstance(body, dict) else None
        status = getattr(error, "status_code", None)
        if type(status) is not int or not 100 <= status <= 599:
            status = None
        request_id = getattr(error, "request_id", None)
        if not (
            isinstance(request_id, str)
            and (
                re.fullmatch(r"req_[A-Za-z0-9_-]{1,96}", request_id)
                or _uuid(request_id)
            )
            and "sk-" not in request_id
        ):
            request_id = None
        correlation = _uuid(operation_key)
        event = {
            "event": "EMAIL_SOURCE_PROVIDER_ERROR",
            "version": 1,
            "operationKey": correlation,
            "httpStatus": status,
            "providerCode": code,
            "providerType": kind,
            "providerRequestId": request_id,
            "classification": (
                _CODE_CLASSES.get(code, "UNKNOWN_429")
                if status == 429
                else "PROVIDER_ERROR"
            ),
            "headers": _headers(getattr(error, "response", None)),
        }
        log.warning("%s", json.dumps(event, separators=(",", ":"), sort_keys=True))
    except Exception:  # noqa: BLE001 - diagnostics must not replace the original outcome.
        return

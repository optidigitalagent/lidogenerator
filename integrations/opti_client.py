"""One-attempt HTTP client for the Opti Lead Import endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

IMPORT_PATH = "/integrations/lead-generator/import-batches"


@dataclass(frozen=True)
class DeliveryError(Exception):
    code: str
    message: str
    retryable: bool

    def __str__(self) -> str:
        return self.message


def normalize_base_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DeliveryError("OPTI_CONFIG_INVALID", "OPTI_BASE_URL must be an HTTP(S) URL", False)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DeliveryError("OPTI_CONFIG_INVALID", "OPTI_BASE_URL contains unsupported components", False)
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc, path, "", ""))


def validate_response(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti returned an invalid response", True)
    batch = data.get("batch")
    items = data.get("items")
    if not isinstance(batch, Mapping) or not isinstance(items, list):
        raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response is missing batch/items", True)
    batch_id = batch.get("id")
    external_batch_id = batch.get("externalBatchId")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(external_batch_id, str):
        raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response has invalid batch identity", True)
    summary: dict[str, Any] = {
        "optiBatchId": batch_id,
        "externalBatchId": external_batch_id,
    }
    for field in ("createdCount", "updatedCount", "duplicateCount", "rejectedCount"):
        value = batch.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DeliveryError("OPTI_RESPONSE_INVALID", f"Opti response has invalid {field}", True)
        summary[field] = value
    bounded_items = []
    if len(items) > 200:
        raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response contains too many items", True)
    for item in items:
        if not isinstance(item, Mapping):
            raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response has an invalid item", True)
        external_id = item.get("externalLeadId")
        outcome = item.get("outcome")
        if not isinstance(external_id, str) or not external_id or len(external_id) > 500 or outcome not in {
            "CREATED", "UPDATED", "POSSIBLE_DUPLICATE", "REJECTED"
        }:
            raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response item shape is invalid", True)
        review_required = item.get("reviewRequired", False)
        if not isinstance(review_required, bool):
            raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response reviewRequired is invalid", True)
        for identifier in (item.get("businessId"), item.get("opportunityId")):
            if identifier is not None and (not isinstance(identifier, str) or len(identifier) > 500):
                raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response item ID is invalid", True)
        bounded_items.append(
            {
                "externalLeadId": external_id[:500],
                "outcome": outcome,
                "businessId": item.get("businessId") if isinstance(item.get("businessId"), str) else None,
                "opportunityId": item.get("opportunityId") if isinstance(item.get("opportunityId"), str) else None,
                "reviewRequired": review_required,
            }
        )
    if sum(summary[field] for field in (
        "createdCount", "updatedCount", "duplicateCount", "rejectedCount"
    )) != len(items):
        raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti response counts do not match items", True)
    summary["items"] = bounded_items[:50]
    summary["receivedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return summary


class OptiClient:
    """Send exactly one request; retry policy belongs exclusively to the outbox."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.token = token.strip()
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        if not self.base_url or not self.token:
            raise DeliveryError(
                "OPTI_CONFIG_MISSING",
                "Opti bridge requires OPTI_BASE_URL and OPTI_IMPORT_TOKEN",
                False,
            )

    async def import_batch(self, payload_bytes: bytes, key: str) -> dict[str, Any]:
        endpoint = urljoin(f"{self.base_url}/", IMPORT_PATH.lstrip("/"))
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    endpoint,
                    content=payload_bytes,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Idempotency-Key": key,
                        "Content-Type": "application/json",
                    },
                )
        except httpx.HTTPError as error:
            raise DeliveryError("OPTI_TRANSPORT_ERROR", type(error).__name__, True) from error
        if response.status_code in {200, 201}:
            try:
                data = response.json()
            except json.JSONDecodeError as error:
                raise DeliveryError("OPTI_RESPONSE_INVALID", "Opti returned invalid JSON", True) from error
            summary = validate_response(data)
            try:
                request_batch_id = json.loads(payload_bytes)["externalBatchId"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DeliveryError("OPTI_REQUEST_INVALID", "Local payload is invalid", False) from error
            if summary["externalBatchId"] != request_batch_id:
                raise DeliveryError(
                    "OPTI_RESPONSE_INVALID", "Opti response batch does not match request", True
                )
            return summary
        if response.status_code in {401, 403}:
            raise DeliveryError("OPTI_UNAUTHORIZED", "Opti rejected bridge credentials", False)
        if response.status_code == 409:
            raise DeliveryError("OPTI_PAYLOAD_CONFLICT", "Opti rejected a conflicting batch", False)
        if 300 <= response.status_code < 400:
            raise DeliveryError("OPTI_REDIRECT_REJECTED", "Opti redirect was rejected", False)
        if response.status_code >= 500 or response.status_code in {408, 425, 429}:
            raise DeliveryError("OPTI_TEMPORARY_ERROR", f"Opti HTTP {response.status_code}", True)
        raise DeliveryError("OPTI_REQUEST_REJECTED", f"Opti HTTP {response.status_code}", False)

import asyncio
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

import config
import db
from integrations import opti_outbox
from integrations.opti_client import DeliveryError, OptiClient, validate_response
from integrations.opti_contract import build_payload
from models import Business


NOW = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)


def make_payload(batch_id="batch-1"):
    return build_payload(
        {
            "external_batch_id": batch_id,
            "niche": "dentistry",
            "city": "Kyiv",
            "count": 50,
            "created_at": "2026-08-08T09:00:00Z",
            "finished_at": "2026-08-08T10:00:00Z",
        },
        [
            Business(
                name="Bella Dent", city="Kyiv", phone="+3801", instagram_url="https://instagram.com/bella",
                site_quality="none", collected_at="2026-08-08T09:20:00Z",
            )
        ],
        generated_at=NOW,
    )


def success_response(payload):
    external_id = payload["leads"][0]["externalLeadId"]
    return {
        "batch": {
            "id": "00000000-0000-0000-0000-000000000001",
            "externalBatchId": payload["externalBatchId"],
            "createdCount": 1,
            "updatedCount": 0,
            "duplicateCount": 0,
            "rejectedCount": 0,
        },
        "items": [{
            "externalLeadId": external_id,
            "outcome": "CREATED",
            "businessId": "b1",
            "opportunityId": "o1",
            "reviewRequired": False,
        }],
    }


class _FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def import_batch(self, payload_bytes, key):
        self.calls.append((payload_bytes, key))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class OutboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(config, "DB_PATH", str(Path(self.temp.name) / "test.db"))
        self.db_patch.start()
        db.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def test_enqueue_is_durable_and_idempotent(self):
        first = opti_outbox.enqueue(make_payload(), now=NOW)
        second = opti_outbox.enqueue(make_payload(), now=NOW)
        self.assertEqual(first["id"], second["id"])
        with db._connect() as conn:
            self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM opti_sync_outbox").fetchone()[0])
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE opti_sync_outbox SET payloadJson = '{}' WHERE id = ?",
                    (first["id"],),
                )

    async def test_disabled_makes_zero_calls(self):
        opti_outbox.enqueue(make_payload(), now=NOW)
        fake = _FakeClient([validate_response(success_response(make_payload()))])
        with patch.object(config, "OPTI_BRIDGE_ENABLED", False):
            self.assertEqual(0, await opti_outbox.deliver_due(client_factory=lambda: fake))
        self.assertEqual([], fake.calls)

    async def test_immediate_success_marks_sent_and_persists_summary(self):
        payload = make_payload()
        opti_outbox.enqueue(payload, now=NOW)
        fake = _FakeClient([validate_response(success_response(payload))])
        with patch.object(config, "OPTI_BRIDGE_ENABLED", True):
            self.assertEqual(1, await opti_outbox.deliver_due(now=NOW, client_factory=lambda: fake))
        row = opti_outbox.get_by_batch("batch-1")
        self.assertEqual("SENT", row["status"])
        self.assertEqual("00000000-0000-0000-0000-000000000001", row["optiBatchId"])
        self.assertEqual(1, json.loads(row["responseSummaryJson"])["createdCount"])

    async def test_transport_and_5xx_retry_with_same_bytes_and_key(self):
        payload = make_payload()
        row = opti_outbox.enqueue(payload, now=NOW)
        fake = _FakeClient([
            DeliveryError("OPTI_TRANSPORT_ERROR", "timeout", True),
            validate_response(success_response(payload)),
        ])
        with patch.object(config, "OPTI_BRIDGE_ENABLED", True), patch.object(
            config, "OPTI_SYNC_RETRY_BASE_SECONDS", 30
        ):
            await opti_outbox.deliver_due(now=NOW, client_factory=lambda: fake)
            retry = opti_outbox.get_by_batch("batch-1")
            self.assertEqual("RETRY", retry["status"])
            self.assertEqual("2026-08-08T10:00:30Z", retry["nextAttemptAt"])
            await opti_outbox.deliver_due(
                now=NOW + timedelta(seconds=30), client_factory=lambda: fake
            )
        self.assertEqual(fake.calls[0], fake.calls[1])
        self.assertEqual(row["idempotencyKey"], fake.calls[0][1])

    async def test_401_and_409_are_failed_and_redacted(self):
        for batch_id, code in (("unauthorized", "OPTI_UNAUTHORIZED"), ("conflict", "OPTI_PAYLOAD_CONFLICT")):
            opti_outbox.enqueue(make_payload(batch_id), now=NOW)
            fake = _FakeClient([DeliveryError(code, "Bearer secret-token rejected", False)])
            with patch.object(config, "OPTI_BRIDGE_ENABLED", True), patch.object(
                config, "OPTI_IMPORT_TOKEN", "secret-token"
            ):
                await opti_outbox.deliver_due(
                    external_batch_id=batch_id, now=NOW, client_factory=lambda: fake
                )
            row = opti_outbox.get_by_batch(batch_id)
            self.assertEqual("FAILED", row["status"])
            self.assertNotIn("secret-token", row["lastErrorMessage"])

    async def test_enabled_missing_config_fails_without_raising(self):
        opti_outbox.enqueue(make_payload(), now=NOW)
        with patch.object(config, "OPTI_BRIDGE_ENABLED", True), patch.object(
            config, "OPTI_BASE_URL", ""
        ), patch.object(config, "OPTI_IMPORT_TOKEN", ""):
            self.assertEqual(0, await opti_outbox.deliver_due(now=NOW))
        self.assertEqual("FAILED", opti_outbox.get_by_batch("batch-1")["status"])

    def test_stale_sending_recovered_and_claim_is_exclusive(self):
        opti_outbox.enqueue(make_payload(), now=NOW)
        first = opti_outbox._claim_due(now=NOW)
        self.assertIsNotNone(first)
        self.assertIsNone(opti_outbox._claim_due(now=NOW))
        self.assertEqual(
            1,
            opti_outbox.recover_stale_sending(
                now=NOW + timedelta(minutes=6), stale_after_seconds=300
            ),
        )
        self.assertEqual("RETRY", opti_outbox.get_by_batch("batch-1")["status"])

    async def test_max_attempts_and_bounded_backoff(self):
        opti_outbox.enqueue(make_payload(), now=NOW)
        fake = _FakeClient([DeliveryError("TEMP", "down", True)] * 3)
        with patch.object(config, "OPTI_BRIDGE_ENABLED", True), patch.object(
            config, "OPTI_SYNC_MAX_ATTEMPTS", 2
        ), patch.object(config, "OPTI_SYNC_RETRY_BASE_SECONDS", 3600):
            await opti_outbox.deliver_due(now=NOW, client_factory=lambda: fake)
            row = opti_outbox.get_by_batch("batch-1")
            self.assertEqual("2026-08-08T11:00:00Z", row["nextAttemptAt"])
            await opti_outbox.deliver_due(
                now=NOW + timedelta(hours=1), client_factory=lambda: fake
            )
        row = opti_outbox.get_by_batch("batch-1")
        self.assertEqual("FAILED", row["status"])
        self.assertEqual(2, row["attempts"])

    async def test_worker_does_not_start_twice(self):
        worker = opti_outbox.OutboxWorker(poll_seconds=60)
        with patch.object(config, "OPTI_BRIDGE_ENABLED", False):
            self.assertTrue(worker.start())
            self.assertFalse(worker.start())
            await worker.stop()


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_success_and_headers(self):
        payload = make_payload()
        observed = {}

        async def handler(request):
            observed["authorization"] = request.headers["Authorization"]
            observed["key"] = request.headers["Idempotency-Key"]
            observed["body"] = request.content
            return httpx.Response(201, json=success_response(payload))

        client = OptiClient(
            "https://opti.example/", "fake-token", 2, transport=httpx.MockTransport(handler)
        )
        body = json.dumps(payload).encode()
        summary = await client.import_batch(body, "stable-key")
        self.assertEqual(1, summary["createdCount"])
        self.assertEqual("Bearer fake-token", observed["authorization"])
        self.assertEqual("stable-key", observed["key"])
        self.assertEqual(body, observed["body"])

    async def test_redirect_is_not_followed(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(307, headers={"Location": "https://evil.example"})

        client = OptiClient("https://opti.example", "fake", 2, transport=httpx.MockTransport(handler))
        with self.assertRaises(DeliveryError) as caught:
            await client.import_batch(b"{}", "key")
        self.assertEqual("OPTI_REDIRECT_REJECTED", caught.exception.code)
        self.assertEqual(1, calls)

    async def test_5xx_is_retryable(self):
        async def handler(request):
            return httpx.Response(503)

        client = OptiClient("https://opti.example", "fake", 2, transport=httpx.MockTransport(handler))
        with self.assertRaises(DeliveryError) as caught:
            await client.import_batch(b"{}", "key")
        self.assertEqual("OPTI_TEMPORARY_ERROR", caught.exception.code)
        self.assertTrue(caught.exception.retryable)

    async def test_invalid_response_shape_is_rejected(self):
        async def handler(request):
            return httpx.Response(200, json={"batch": {}})

        client = OptiClient("https://opti.example", "fake", 2, transport=httpx.MockTransport(handler))
        with self.assertRaises(DeliveryError) as caught:
            await client.import_batch(b"{}", "key")
        self.assertEqual("OPTI_RESPONSE_INVALID", caught.exception.code)

    async def test_response_batch_must_match_request(self):
        payload = make_payload()
        response = success_response(payload)
        response["batch"]["externalBatchId"] = "different-batch"

        async def handler(request):
            return httpx.Response(200, json=response)

        client = OptiClient("https://opti.example", "fake", 2, transport=httpx.MockTransport(handler))
        with self.assertRaises(DeliveryError) as caught:
            await client.import_batch(json.dumps(payload).encode(), "key")
        self.assertEqual("OPTI_RESPONSE_INVALID", caught.exception.code)

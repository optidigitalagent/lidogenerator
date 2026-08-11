import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import TestClient, TestServer

import config
import db
from control_api import create_app, search_dto
from integrations import opti_outbox
from search_runtime import SearchRuntime

TOKEN = "control-token-that-is-definitely-32-characters"


class ControlApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            config, "DB_PATH", str(Path(self.temp.name) / "test.db")
        )
        self.db_patch.start()
        db.init_db()
        self.runtime = SearchRuntime(2)

        async def waiting_runner(task_id, *, progress_callback, stop_event):
            await stop_event.wait()
            db.update_task_progress(task_id, {"stage": "stopped", "targetLeads": 1})
            db.update_task_status(task_id, "stopped")

        self.runner_patch = patch(
            "search_runtime.orchestrator.run_search", waiting_runner
        )
        self.runner_patch.start()
        self.client = TestClient(TestServer(create_app(self.runtime, TOKEN)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.runtime.shutdown()
        await self.client.close()
        self.runner_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def auth(self):
        return {"Authorization": f"Bearer {TOKEN}"}

    async def test_health_public_and_bearer_auth(self):
        response = await self.client.get("/health")
        self.assertEqual(200, response.status)
        self.assertNotIn(TOKEN, await response.text())
        response = await self.client.get("/internal/opti/v1/searches?token=" + TOKEN)
        self.assertEqual(401, response.status)
        self.assertEqual("AUTH_REQUIRED", (await response.json())["error"]["code"])
        response = await self.client.get(
            "/internal/opti/v1/searches", headers={"Authorization": "Bearer wrong"}
        )
        self.assertEqual(
            "INVALID_CONTROL_TOKEN", (await response.json())["error"]["code"]
        )
        self.assertNotIn(TOKEN, await response.text())

    async def test_create_idempotency_list_get_stop_and_redaction(self):
        headers = {**self.auth(), "Idempotency-Key": "attempt-1"}
        payload = {"niche": " dentistry ", "city": " Kyiv ", "targetLeads": 1}
        response = await self.client.post(
            "/internal/opti/v1/searches", headers=headers, json=payload
        )
        self.assertEqual(201, response.status)
        created = await response.json()
        self.assertEqual("OPTI", created["initiatedVia"])
        self.assertEqual("dentistry", created["niche"])
        search_id = created["searchId"]
        replay = await self.client.post(
            "/internal/opti/v1/searches", headers=headers, json=payload
        )
        self.assertEqual(200, replay.status)
        self.assertEqual(search_id, (await replay.json())["searchId"])
        conflict = await self.client.post(
            "/internal/opti/v1/searches",
            headers=headers,
            json={**payload, "targetLeads": 2},
        )
        self.assertEqual(409, conflict.status)
        listed = await (
            await self.client.get(
                "/internal/opti/v1/searches?limit=50", headers=self.auth()
            )
        ).json()
        self.assertEqual([search_id], [item["searchId"] for item in listed["searches"]])
        dto_text = json.dumps(listed)
        for forbidden in ("attempt-1", "csv_path", "payloadJson", "payloadHash"):
            self.assertNotIn(forbidden, dto_text)
        found = await self.client.get(
            f"/internal/opti/v1/searches/{search_id}", headers=self.auth()
        )
        self.assertEqual(200, found.status)
        stopped = await self.client.post(
            f"/internal/opti/v1/searches/{search_id}/stop", headers=self.auth()
        )
        self.assertEqual(200, stopped.status)
        await asyncio.sleep(0)
        self.assertEqual("stopped", db.get_task_by_search_id(search_id)["status"])

    async def test_strict_validation_limits_and_not_found(self):
        headers = {**self.auth(), "Idempotency-Key": "attempt-2"}
        response = await self.client.post(
            "/internal/opti/v1/searches",
            headers=headers,
            json={"niche": "x", "city": "y", "targetLeads": 1, "extra": 1},
        )
        self.assertEqual(400, response.status)
        response = await self.client.get(
            "/internal/opti/v1/searches?limit=101", headers=self.auth()
        )
        self.assertEqual(400, response.status)
        response = await self.client.get(
            "/internal/opti/v1/searches/missing", headers=self.auth()
        )
        self.assertEqual(404, response.status)

    async def test_failed_retry_changes_only_requested_batch_and_sent_is_idempotent(
        self,
    ):
        with patch.object(config, "OPTI_BRIDGE_ENABLED", True):
            ids = []
            for name in ("one", "two"):
                task_id = db.create_task(name, "Kyiv", 1, initiated_via="OPTI")
                db.update_task_status(task_id, "done")
                search_id = db.get_task(task_id)["external_batch_id"]
                ids.append(search_id)
                with db._connect() as conn:
                    conn.execute(
                        "INSERT INTO opti_sync_outbox (externalBatchId,schemaVersion,payloadJson,payloadHash,idempotencyKey,status,attempts,createdAt,updatedAt) VALUES (?,?,?,?,?,'FAILED',3,?,?)",
                        (
                            search_id,
                            "v",
                            "{}",
                            "0" * 64,
                            "key-" + name,
                            "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z",
                        ),
                    )
            with patch.object(
                opti_outbox, "deliver_due", AsyncMock(return_value=0)
            ) as deliver:
                response = await self.client.post(
                    f"/internal/opti/v1/searches/{ids[0]}/retry-sync",
                    headers=self.auth(),
                )
            self.assertEqual(200, response.status)
            deliver.assert_awaited_once_with(external_batch_id=ids[0], limit=1)
            self.assertEqual("RETRY", opti_outbox.get_by_batch(ids[0])["status"])
            self.assertEqual("FAILED", opti_outbox.get_by_batch(ids[1])["status"])


class SearchDtoTests(unittest.TestCase):
    def test_defensive_summary_and_no_raw_fields(self):
        task = {
            "external_batch_id": "search-1",
            "niche": "n",
            "city": "c",
            "count": 1,
            "status": "done",
            "initiated_via": "TELEGRAM",
            "created_at": "now",
            "updated_at": "now",
            "finished_at": "now",
            "progress_json": '{"rawToken":"must-not-escape"}',
            "last_error_code": None,
            "last_error_message": None,
        }
        with patch.object(opti_outbox, "get_by_batch", return_value=None):
            dto = search_dto(task)
        self.assertEqual("NOT_ENQUEUED", dto["sync"]["status"])
        self.assertEqual({"stage": "done", "targetLeads": 1}, dto["progress"])
        self.assertNotIn("control_idempotency_key", dto)
        self.assertNotIn("rawToken", dto["progress"])

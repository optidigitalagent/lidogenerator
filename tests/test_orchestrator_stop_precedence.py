"""Regression tests for user-stop precedence over provider failures."""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import config
import db
import orchestrator


class OrchestratorStopPrecedenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "stop-precedence.db"
        self.db_patch = patch.object(config, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.db_patch.stop()
        self.temp.cleanup()

    async def _run_provider_failure(
        self,
        error: Exception,
        *,
        request_stop: bool,
    ) -> tuple[int, object, list[str], AsyncMock]:
        task_id = db.create_task("test niche", "Test City", 1, initiated_via="OPTI")
        stop_event = asyncio.Event()
        provider_in_flight = asyncio.Event()
        release_provider = asyncio.Event()
        messages: list[str] = []
        bridge = AsyncMock(return_value="")

        async def progress(message: str) -> None:
            messages.append(message)

        async def failing_collect_stream(*args, **kwargs):
            provider_in_flight.set()
            await release_provider.wait()
            raise error
            yield  # pragma: no cover - keeps this an async generator

        with (
            patch.object(config, "WEBSITE_RESOLVER_MODE", "off"),
            patch.object(config, "INSTAGRAM_FIRST_PARTY_MODE", "off"),
            patch.object(config, "RENDERED_SITE_AUDIT_MODE", "off"),
            patch.object(orchestrator.collector, "collect_stream", new=failing_collect_stream),
            patch.object(orchestrator, "finalize_completed_task", new=bridge),
        ):
            search = asyncio.create_task(
                orchestrator.run_search(
                    task_id,
                    progress_callback=progress,
                    stop_event=stop_event,
                    progress_interval=0,
                )
            )
            await asyncio.wait_for(provider_in_flight.wait(), timeout=1)
            if request_stop:
                stop_event.set()
            release_provider.set()
            result = await search

        return task_id, result, messages, bridge

    def _task_progress(self, task_id: int) -> tuple[object, dict]:
        task = db.get_task(task_id)
        self.assertIsNotNone(task)
        return task, json.loads(task["progress_json"])

    def _assert_no_outbox_rows(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM opti_sync_outbox"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(0, count)

    async def test_timeout_after_in_flight_stop_finishes_stopped(self) -> None:
        task_id, result, messages, bridge = await self._run_provider_failure(
            TimeoutError("provider timeout"),
            request_stop=True,
        )

        task, progress = self._task_progress(task_id)
        self.assertIsNone(result)
        self.assertEqual("stopped", task["status"])
        self.assertEqual("stopped", progress["stage"])
        self.assertEqual("USER_STOPPED", progress["stopReason"])
        self.assertIsNotNone(task["finished_at"])
        self.assertIsNone(task["last_error_code"])
        self.assertIsNone(task["last_error_message"])
        self.assertIn("Пошук зупинено", messages[-1])
        self.assertNotIn("Помилка", messages[-1])
        bridge.assert_not_awaited()
        self._assert_no_outbox_rows()

    async def test_generic_provider_error_after_stop_finishes_stopped(self) -> None:
        task_id, result, _, bridge = await self._run_provider_failure(
            RuntimeError("provider exploded"),
            request_stop=True,
        )

        task, progress = self._task_progress(task_id)
        self.assertIsNone(result)
        self.assertEqual("stopped", task["status"])
        self.assertEqual("stopped", progress["stage"])
        self.assertEqual("USER_STOPPED", progress["stopReason"])
        self.assertIsNone(task["last_error_code"])
        self.assertIsNone(task["last_error_message"])
        bridge.assert_not_awaited()

    async def test_provider_timeout_without_stop_remains_error(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "provider timeout"):
            await self._run_provider_failure(
                TimeoutError("provider timeout"),
                request_stop=False,
            )

        task = db.get_task(1)
        self.assertIsNotNone(task)
        progress = json.loads(task["progress_json"])
        self.assertEqual("error", task["status"])
        self.assertEqual("error", progress["stage"])
        self.assertEqual("TIMEOUTERROR", task["last_error_code"])
        self.assertEqual("provider timeout", task["last_error_message"])

    async def test_explicit_search_stopped_does_not_call_bridge(self) -> None:
        task_id, result, _, bridge = await self._run_provider_failure(
            orchestrator.SearchStopped(),
            request_stop=False,
        )

        task, progress = self._task_progress(task_id)
        self.assertIsNone(result)
        self.assertEqual("stopped", task["status"])
        self.assertEqual("stopped", progress["stage"])
        self.assertEqual("USER_STOPPED", progress["stopReason"])
        bridge.assert_not_awaited()
        self._assert_no_outbox_rows()


if __name__ == "__main__":
    unittest.main()

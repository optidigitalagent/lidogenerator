import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bot
import config
import db
from search_runtime import SearchRuntime


class SearchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            config, "DB_PATH", str(Path(self.temp.name) / "test.db")
        )
        self.db_patch.start()
        db.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    async def test_queued_stop_is_immediate_and_engine_never_starts(self):
        runtime = SearchRuntime(1)
        running = 0
        maximum = 0
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        invoked: list[int] = []

        async def runner(task_id, *, progress_callback, stop_event):
            nonlocal running, maximum
            invoked.append(task_id)
            running += 1
            maximum = max(maximum, running)
            try:
                if task_id == first:
                    first_started.set()
                    await release_first.wait()
                    db.update_task_status(task_id, "done")
            finally:
                running -= 1

        first = db.create_task("one", "Kyiv", 1, initiated_via="TELEGRAM")
        second = db.create_task("two", "Kyiv", 1, initiated_via="OPTI")
        self.assertTrue(runtime.launch(first, run_search=runner))
        self.assertFalse(runtime.launch(first, run_search=runner))
        self.assertTrue(runtime.launch(second, run_search=runner))
        await first_started.wait()
        await asyncio.sleep(0)
        self.assertEqual([first], invoked)

        self.assertTrue(runtime.stop(second))
        row = db.get_task(second)
        progress = json.loads(row["progress_json"])
        self.assertEqual("stopped", row["status"])
        self.assertEqual("stopped", progress["stage"])
        self.assertEqual("USER_STOPPED", progress["stopReason"])
        self.assertIsNotNone(row["finished_at"])
        self.assertFalse(runtime.is_active(second))
        self.assertEqual(1, runtime.active_count)
        self.assertFalse(runtime.launch(second, run_search=runner))

        release_first.set()
        await runtime.shutdown()
        self.assertEqual([first], invoked)
        self.assertEqual("stopped", db.get_task(second)["status"])
        self.assertEqual(1, maximum)
        self.assertEqual(0, runtime.active_count)

    async def test_running_stop_waits_for_runner_acknowledgement(self):
        runtime = SearchRuntime(1)
        started = asyncio.Event()
        stop_observed = asyncio.Event()
        acknowledge = asyncio.Event()

        async def runner(task_id, *, progress_callback, stop_event):
            started.set()
            await stop_event.wait()
            stop_observed.set()
            await acknowledge.wait()
            db.update_task_progress(
                task_id, {"stage": "stopped", "stopReason": "USER_STOPPED"}
            )
            db.update_task_status(task_id, "stopped")

        task_id = db.create_task("one", "Kyiv", 1)
        self.assertTrue(runtime.launch(task_id, run_search=runner))
        await started.wait()
        self.assertTrue(runtime.stop(task_id))
        await stop_observed.wait()

        self.assertEqual("new", db.get_task(task_id)["status"])
        self.assertTrue(runtime.is_active(task_id))
        acknowledge.set()
        await runtime.shutdown()
        self.assertEqual("stopped", db.get_task(task_id)["status"])
        self.assertFalse(runtime.is_active(task_id))

    async def test_queued_telegram_stop_releases_user_for_next_search(self):
        runtime = SearchRuntime(1)
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def runner(task_id, *, progress_callback, stop_event):
            first_started.set()
            await release_first.wait()

        first = db.create_task("one", "Kyiv", 1, initiated_via="OPTI")
        queued = db.create_task("two", "Kyiv", 1, initiated_via="TELEGRAM")
        runtime.launch(first, run_search=runner)
        runtime.launch(queued, run_search=runner)
        await first_started.wait()
        self.assertTrue(runtime.stop(queued))

        user_id = 123
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        with (
            patch.object(bot, "SEARCH_RUNTIME", runtime),
            patch.object(bot, "ACTIVE", {user_id: queued}),
            patch.object(bot, "is_allowed", return_value=True),
        ):
            state = await bot.cmd_search(update, SimpleNamespace())

        self.assertEqual(bot.NICHE, state)
        update.message.reply_text.assert_awaited_once()
        release_first.set()
        await runtime.shutdown()

    async def test_restart_recovery_is_sanitized(self):
        first = db.create_task("one", "Kyiv", 1)
        second = db.create_task("two", "Kyiv", 1)
        db.update_task_status(second, "done")
        self.assertEqual(1, db.recover_interrupted_tasks())
        row = db.get_task(first)
        self.assertEqual("error", row["status"])
        self.assertEqual("PROCESS_RESTARTED", row["last_error_code"])
        self.assertEqual(
            "Search was interrupted by service restart", row["last_error_message"]
        )
        self.assertEqual("done", db.get_task(second)["status"])

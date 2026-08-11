import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    async def test_shared_semaphore_duplicate_stop_and_shutdown(self):
        runtime = SearchRuntime(1)
        running = 0
        maximum = 0
        started = [asyncio.Event(), asyncio.Event()]

        async def runner(task_id, *, progress_callback, stop_event):
            nonlocal running, maximum
            index = 0 if task_id == first else 1
            running += 1
            maximum = max(maximum, running)
            started[index].set()
            try:
                await stop_event.wait()
                db.update_task_status(task_id, "stopped")
                return
            finally:
                running -= 1

        first = db.create_task("one", "Kyiv", 1, initiated_via="TELEGRAM")
        second = db.create_task("two", "Kyiv", 1, initiated_via="OPTI")
        self.assertTrue(runtime.launch(first, run_search=runner))
        self.assertFalse(runtime.launch(first, run_search=runner))
        self.assertTrue(runtime.launch(second, run_search=runner))
        await started[0].wait()
        self.assertTrue(runtime.stop(first))
        await started[1].wait()
        self.assertTrue(runtime.stop(second))
        await runtime.shutdown()
        self.assertEqual(1, maximum)
        self.assertEqual(0, runtime.active_count)

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

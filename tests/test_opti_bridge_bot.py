import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import config
import db
from integrations import opti_outbox
from integrations.opti_bridge import finalize_completed_task
from models import Business


class BridgeFinalizationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(config, "DB_PATH", str(Path(self.temp.name) / "test.db"))
        self.db_patch.start()
        db.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    async def test_remote_failure_does_not_break_completed_result(self):
        task_id = db.create_task("dentistry", "Kyiv", 50)
        db.save_businesses([
            Business(
                task_id=task_id,
                name="Final lead",
                city="Kyiv",
                phone="+3801",
                instagram_url="https://instagram.com/final",
                site_quality="none",
            )
        ])
        db.update_task_status(task_id, "done", csv_path="exports/result.xlsx")
        with patch.object(
            opti_outbox, "deliver_due", AsyncMock(side_effect=RuntimeError("Opti down"))
        ):
            message = await finalize_completed_task(task_id)
        self.assertEqual("Opti sync pending — /sync", message)
        self.assertEqual("done", db.get_task(task_id)["status"])
        self.assertEqual(1, opti_outbox.summary_counts()["PENDING"])

    async def test_stopped_partial_set_is_not_enqueued(self):
        task_id = db.create_task("dentistry", "Kyiv", 50)
        db.save_businesses([
            Business(task_id=task_id, name="Partial", city="Kyiv", phone="+1")
        ])
        db.update_task_status(task_id, "stopped")
        self.assertEqual("", await finalize_completed_task(task_id))
        self.assertEqual(0, sum(opti_outbox.summary_counts().values()))


class SyncCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_preserves_access_control(self):
        import bot

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=22),
            effective_chat=SimpleNamespace(id=33),
            message=SimpleNamespace(reply_text=AsyncMock()),
            callback_query=None,
        )
        with patch.object(config, "ALLOWED_USER_IDS", {11}), patch.object(
            bot, "deny", AsyncMock()
        ) as deny:
            await bot.cmd_sync(update, SimpleNamespace())
        deny.assert_awaited_once()
        update.message.reply_text.assert_not_awaited()

    async def test_sync_message_has_summary_not_payload_or_token(self):
        import bot

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=11),
            effective_chat=SimpleNamespace(id=33),
            message=SimpleNamespace(reply_text=AsyncMock()),
            callback_query=None,
        )
        with patch.object(config, "ALLOWED_USER_IDS", {11}), patch.object(
            config, "OPTI_IMPORT_TOKEN", "super-secret"
        ), patch.object(
            bot.opti_bridge,
            "reconcile_completed_tasks",
            return_value={"examined": 0, "enqueued": 0, "errors": 0},
        ) as reconcile, patch.object(
            bot.opti_outbox, "retry_failed", return_value=0
        ), patch.object(
            bot.opti_outbox, "deliver_due", AsyncMock(return_value=0)
        ), patch.object(
            bot.opti_outbox, "format_summary", return_value="Opti sync: pending 1"
        ):
            await bot.cmd_sync(update, SimpleNamespace())
        reconcile.assert_called_once_with(limit=100)
        text = update.message.reply_text.await_args.args[0]
        self.assertIn("pending 1", text)
        self.assertNotIn("super-secret", text)
        self.assertNotIn("payload", text.casefold())

    async def test_post_init_recovers_then_reconciles_before_starting_worker(self):
        import bot

        calls = []
        with patch.object(
            bot.opti_outbox,
            "recover_stale_sending",
            side_effect=lambda **_kwargs: calls.append("recover"),
        ), patch.object(
            bot.opti_bridge,
            "reconcile_completed_tasks",
            side_effect=lambda **_kwargs: calls.append("reconcile"),
        ), patch.object(
            bot.OUTBOX_WORKER,
            "start",
            side_effect=lambda: calls.append("start"),
        ):
            await bot._start_outbox_worker(SimpleNamespace())
        self.assertEqual(["recover", "reconcile", "start"], calls)

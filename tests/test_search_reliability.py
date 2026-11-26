"""Original target20/history-only incident through real runtime, SQLite and HTTP.

Only Maps is synthetic. No provider request, history reset or fake terminal state.
"""

import asyncio
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

import candidate_history as history
import config
import db
import orchestrator
from control_api import create_app
from models import Business
from search_runtime import SearchRuntime


class SearchReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_target20_history_exhaustion_is_observable_and_releases_slot(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            settings = {
                "DB_PATH": str(Path(directory) / "incident.sqlite"),
                "EXPORT_DIR": Path(directory) / "exports",
                "CANDIDATE_HISTORY_MODE": "apply",
                "DEEP_DISCOVERY_MODE": "off",
                "LEAD_WEBSITE_POLICY": "verified_no_site_only",
                "WEBSITE_PRESENCE_VERIFICATION_MODE": "apply",
                "WEBSITE_RESOLVER_MODE": "off",
                "INSTAGRAM_FIRST_PARTY_MODE": "off",
                "RENDERED_SITE_AUDIT_MODE": "off",
                "OPTI_BRIDGE_ENABLED": False,
            }
            for name, value in settings.items():
                stack.enter_context(patch.object(config, name, value))
            db.init_db()
            scope = history.canonical_scope_key("стоматологія", "винница")
            candidates = [
                Business(
                    name=f"Fixture clinic {i}",
                    niche="стоматологія",
                    city="Вінниця",
                    address=f"Fixture street {i}",
                    phone=f"+38050{i:07d}",
                    google_place_id=f"r21-place-{i}",
                    instagram_url=f"https://instagram.com/fixture_{i}",
                )
                for i in range(138)
            ]
            for item in candidates:
                basis, key = history.candidate_fingerprint(item, "Вінниця")
                history.claim_candidate(scope, key, basis, 1)
                history.mark_candidate_checked(scope, key, 1, "website_present_search")

            observed = asyncio.Event()
            release = asyncio.Event()
            streams = 0

            async def source(*args, progress_callback=None, **kwargs):
                nonlocal streams
                stream = streams
                streams += 1
                chunk = candidates[stream * 20 : (stream + 1) * 20]
                opened = len(chunk) + (3 if stream == 6 else 0)
                for i in range(opened):
                    kwargs["discovery_session"].record_card_opened()
                    await progress_callback(i + 1)
                if chunk:
                    yield chunk
                if stream == 6:
                    observed.set()
                    await release.wait()

            class NoProvider:
                async def search(self, request):
                    raise AssertionError("Checked history must never call the provider")

            stack.enter_context(
                patch.object(orchestrator.collector, "collect_stream", source)
            )
            stack.enter_context(
                patch.object(
                    orchestrator,
                    "build_configured_search_provider",
                    return_value=NoProvider(),
                )
            )
            runtime = SearchRuntime(3)
            token = "r21-offline-control-token-over-32-characters"
            client = TestClient(TestServer(create_app(runtime, token)))
            await client.start_server()
            headers = {
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "incident",
            }
            try:
                response = await client.post(
                    "/internal/opti/v1/searches",
                    headers=headers,
                    json={
                        "niche": "стоматологія",
                        "city": "винница",
                        "targetLeads": 20,
                        "searchMode": "WEBSITE_NEED",
                    },
                )
                self.assertEqual(201, response.status)
                search_id = (await response.json())["searchId"]
                await asyncio.wait_for(observed.wait(), 5)
                path = "/internal/opti/v1/searches/" + search_id
                live = await (await client.get(path, headers=headers)).json()
                self.assertEqual("checking", live["status"])
                self.assertEqual(141, live["progress"]["visitedBusinesses"])
                self.assertEqual(138, live["progress"]["skippedPreviouslyChecked"])
                self.assertEqual(0, live["progress"]["checkedCandidates"])
                self.assertEqual(1, runtime.active_count)
                release.set()
                for _ in range(200):
                    if runtime.active_count == 0:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(0, runtime.active_count)
                final = await (await client.get(path, headers=headers)).json()
                self.assertEqual("done", final["status"])
                self.assertEqual("queries_exhausted", final["progress"]["stopReason"])
                self.assertEqual(7, final["progress"]["normalQueriesCompleted"])
                self.assertEqual(0, final["progress"]["qualifiedLeads"])
                self.assertEqual("NOT_ENQUEUED", final["sync"]["status"])
                task = db.get_task_by_search_id(search_id)
                self.assertEqual(
                    138, json.loads(task["progress_json"])["skippedPreviouslyChecked"]
                )
                self.assertEqual([], db.get_businesses(task["id"]))
                self.assertEqual(7, streams)
                self.assertEqual(set(), runtime._running)
            finally:
                release.set()
                await runtime.shutdown()
                await client.close()

"""R20 offline test server: real runtime/SQLite/Bridge; synthetic outside sources.

Never run in production. Binds loopback only and refuses non-fixture use.
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
assert os.environ.get("R20_OFFLINE_FIXTURE") == "1"
from aiohttp import web

import db
import orchestrator
from control_api import create_app, search_dto
from models import Business
from search_runtime import SearchRuntime
from website_pipeline import WebsiteAuditStatus

gates = {}
trace = []


async def collect(niche, city, **kwargs):
    task = next(dict(t) for t in db.list_tasks(100) if t["niche"] == niche)
    gate = gates.setdefault(niche, asyncio.Event())
    trace.append(
        {
            "event": "collector_started",
            "searchId": task["external_batch_id"],
            "at": datetime.now(UTC).isoformat(),
        }
    )
    while not gate.is_set():
        if kwargs["stop_flag"]():
            return
        await asyncio.sleep(0.01)
    if niche.startswith("failure"):
        raise RuntimeError("offline source failed")
    if kwargs["progress_callback"]:
        await kwargs["progress_callback"](2)
    yield [
        Business(
            name=f"Клініка Альфа {niche} {i}",
            niche=niche,
            city=city,
            address=f"Fixture street {task['id']} {i}",
            phone=f"+38050{task['id']:05d}{i:02d}",
            google_place_id=f"r20-{task['id']}-{i}",
            google_maps_url=f"https://maps.google.com/?cid={task['id']}0{i}",
            instagram_url=f"https://instagram.com/r20_{task['id']}_{i}",
            website=f"https://weak-{task['id']}.test/" if i else "",
        )
        for i in range(2)
    ]


async def website_source(client, semaphore, business):
    # A synthetic HTTP outcome; the classification and Bridge mappings are real.
    if not business.website:
        business.has_site = False
        business.site_quality = "none"
        orchestrator.site_checker._store_audit(
            business, WebsiteAuditStatus.NO_OFFICIAL_SITE
        )
    else:
        business.has_site = True
        business.site_quality = orchestrator.site_checker._analyze(
            "<frameset><title>Old fixture</title></frameset>", business.website, 0
        )
        orchestrator.site_checker._store_audit(
            business,
            WebsiteAuditStatus.BAD,
            http_status=200,
            final_url=business.website,
        )


async def main():
    db.init_db()
    runtime = SearchRuntime(3)
    app = create_app(runtime, os.environ["FIXTURE_CONTROL_TOKEN"])

    async def release(request):
        gates.setdefault(request.match_info["niche"], asyncio.Event()).set()
        return web.json_response({"released": True})

    async def observation(request):
        return web.json_response(
            {
                "trace": trace,
                "searches": [search_dto(dict(t)) for t in db.list_tasks(100)],
                "running": sorted(runtime._running),
            }
        )

    app.router.add_post("/fixtures/release/{niche}", release)
    app.router.add_get("/fixtures/trace", observation)
    with (
        patch.object(orchestrator.collector, "collect_stream", collect),
        patch.object(orchestrator.site_checker, "_check_one", website_source),
        patch.object(orchestrator.social_checker, "check_instagram", AsyncMock()),
        patch.object(orchestrator.ai_scorer, "score_businesses", AsyncMock()),
    ):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        print(json.dumps({"port": port}), flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            for gate in gates.values():
                gate.set()
            await runtime.shutdown()
            await runner.cleanup()


asyncio.run(main())

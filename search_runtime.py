"""One in-process owner for every Telegram and Opti search."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import db
import orchestrator

ProgressCallback = Callable[[str], Awaitable[None]]
CompletionCallback = Callable[[str | None], Awaitable[None]]

log = logging.getLogger("lead_hunter.search_runtime")


class SearchRuntime:
    def __init__(self, max_concurrent_searches: int) -> None:
        self.max_concurrent_searches = max_concurrent_searches
        self._semaphore = asyncio.Semaphore(max_concurrent_searches)
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._stop_events: dict[int, asyncio.Event] = {}
        self._closing = False

    @property
    def active_count(self) -> int:
        return sum(not task.done() for task in self._tasks.values())

    def is_active(self, task_id: int) -> bool:
        task = self._tasks.get(task_id)
        return task is not None and not task.done()

    def launch(
        self,
        task_id: int,
        *,
        progress_callback: ProgressCallback | None = None,
        completion_callback: CompletionCallback | None = None,
        run_search: Callable[..., Awaitable[str | None]] | None = None,
    ) -> bool:
        if self._closing or self.is_active(task_id):
            return False
        stop_event = asyncio.Event()
        runner = run_search or orchestrator.run_search

        async def run() -> None:
            result: str | None = None
            try:
                async with self._semaphore:
                    if stop_event.is_set():
                        db.update_task_progress(
                            task_id, {"stage": "stopped", "stopReason": "USER_STOPPED"}
                        )
                        db.update_task_status(task_id, "stopped")
                        return
                    result = await runner(
                        task_id,
                        progress_callback=progress_callback,
                        stop_event=stop_event,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                # The orchestrator persists a sanitized error snapshot. This boundary
                # prevents an unobserved task exception from taking down either UI.
                log.exception("Search task %s failed", task_id)
            finally:
                if completion_callback is not None:
                    try:
                        await completion_callback(result)
                    except Exception:
                        log.exception("Search completion callback %s failed", task_id)
                self._tasks.pop(task_id, None)
                self._stop_events.pop(task_id, None)

        self._stop_events[task_id] = stop_event
        self._tasks[task_id] = asyncio.create_task(run(), name=f"search-{task_id}")
        return True

    def stop(self, task_id: int) -> bool:
        event = self._stop_events.get(task_id)
        if event is None:
            return False
        event.set()
        return True

    async def shutdown(self) -> None:
        self._closing = True
        for event in self._stop_events.values():
            event.set()
        pending = tuple(self._tasks.values())
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._stop_events.clear()

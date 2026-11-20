"""Opt-in local transport observations; never record request or response contents."""

from __future__ import annotations

import importlib.metadata
import re
from time import monotonic


class ProviderObservation:
    """A bounded, request-local observer for the existing SDK call.

    The adapter does not create this object for HTTP clients. Local diagnostic
    callers may pass one explicitly and persist ``snapshot()`` themselves.
    httpcore trace info contains headers and streams, so only event names,
    durations and exception *classes* are retained.
    """

    def __init__(self):
        self.started = monotonic()
        self.events = []
        self.active = {}
        self.request_id = None
        self.status = None
        self.cause_classes = []
        self.cancelled = False
        self.timeout = None
        self.completed = False
        self.http_client_modules = []

    def http_client(self):
        from openai import DefaultAsyncHttpxClient

        client = DefaultAsyncHttpxClient(
            event_hooks={"request": [self.on_request], "response": [self.on_response]}
        )
        self.http_client_modules = list(
            dict.fromkeys(cls.__module__ for cls in type(client).__mro__)
        )
        return client

    async def on_request(self, request):
        # Read only numeric timeout settings; do not serialize the request.
        timeout = request.extensions.get("timeout", {})
        self.timeout = {
            name: value if type(value) in (float, int) else None
            for name, value in timeout.items()
            if name in {"connect", "read", "write", "pool"}
        }
        request.extensions["trace"] = self.trace

    async def on_response(self, response):
        self.status = response.status_code
        request_id = response.headers.get("x-request-id", "")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            self.request_id = request_id

    async def trace(self, event, info):
        if not re.fullmatch(
            r"(?:connection|http11|http2)\."
            r"(?:connect_tcp|start_tls|send_request_headers|send_request_body|"
            r"receive_response_headers|receive_response_body|response_closed|close)\."
            r"(?:started|complete|failed)",
            event,
        ):
            return
        now = monotonic()
        phase, state = event.rsplit(".", 1)
        if state == "started":
            self.active[phase] = now
        start = self.active.pop(phase, None) if state != "started" else None
        if len(self.events) < 80:
            self.events.append(
                {
                    "event": event,
                    "elapsedSeconds": round(now - self.started, 6),
                    "phaseSeconds": round(now - start, 6)
                    if start is not None
                    else None,
                    "exceptionClass": type(info["exception"]).__name__
                    if state == "failed" and "exception" in info
                    else None,
                }
            )

    def failure(self, error):
        seen = set()
        while error is not None and id(error) not in seen and len(seen) < 8:
            seen.add(id(error))
            self.cause_classes.append(type(error).__name__)
            error = error.__cause__ or (
                error.__context__ if not error.__suppress_context__ else None
            )

    def snapshot(self):
        versions = {}
        for package in ("openai", "httpx", "httpcore", "httpx2", "httpcore2"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "UNKNOWN"
        return {
            "versions": versions,
            "httpClientModules": self.http_client_modules,
            "sdkTimeoutSeconds": 22,
            "sdkMaxRetries": 0,
            "httpTimeoutSeconds": self.timeout,
            "elapsedSeconds": round(monotonic() - self.started, 6),
            "httpStatus": self.status,
            "requestId": self.request_id,
            "causeClasses": self.cause_classes,
            "cancelled": self.cancelled,
            "completed": self.completed,
            "events": list(self.events),
            "transportWithoutEvents": "UNKNOWN" if not self.events else None,
        }

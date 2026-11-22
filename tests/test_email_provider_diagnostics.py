"""Real installed SDK exceptions, fake SDK transport, no provider/site calls."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import email_provider_diagnostics as diagnostics
import email_source_adapter as adapter

KEY = "11111111-1111-4111-8111-111111111111"
BODY = {"name": "Fixture", "city": "Київ", "mode": "NAME_CITY", "operationKey": KEY}


def sdk_error(
    code="rate_limit_exceeded",
    kind="rate_limit_error",
    status=429,
    headers=None,
    body=None,
):
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        headers=headers or {"x-request-id": "req_fixture"},
    )
    error_class = openai.RateLimitError if status == 429 else openai.APIStatusError
    return error_class(
        "PRIVATE_PROVIDER_BODY_DO_NOT_LOG",
        response=response,
        body={"code": code, "type": kind, "message": "PRIVATE_BODY"}
        if body is None
        else body,
    )


def recorded(error, operation=KEY):
    with patch.object(diagnostics.log, "warning") as logger:
        diagnostics.log_provider_error(error, operation)
        logger.assert_called_once()
        return json.loads(logger.call_args.args[1])


class SafeDiagnostics(unittest.TestCase):
    def test_supported_codes_preserved_separately_and_not_inferred_from_broad_type(
        self,
    ):
        for code, expected in {
            "credit_balance_exhausted": "CREDIT_BALANCE_EXHAUSTED",
            "organization_spend_limit_exceeded": "ORGANIZATION_SPEND_LIMIT_EXCEEDED",
            "project_spend_limit_exceeded": "PROJECT_SPEND_LIMIT_EXCEEDED",
            "organization_usage_limit_exceeded": "ORGANIZATION_USAGE_LIMIT_EXCEEDED",
            "rate_limit_exceeded": "RATE_LIMIT_EXCEEDED",
            "slow_down": "RATE_LIMIT_EXCEEDED",
            "rate_limit_error": "RATE_LIMIT_EXCEEDED",
            "insufficient_quota": "INSUFFICIENT_QUOTA",
        }.items():
            with self.subTest(code=code):
                event = recorded(sdk_error(code, "insufficient_quota"))
                self.assertEqual(code, event["providerCode"])
                self.assertEqual("insufficient_quota", event["providerType"])
                self.assertEqual(expected, event["classification"])
                self.assertEqual(KEY, event["operationKey"])

    def test_absent_malformed_new_codes_stay_unknown_despite_type_or_headers(self):
        for code in [
            None,
            "future_provider_code",
            {},
            [],
            42,
            True,
            "RATE_LIMIT_EXCEEDED",
            "x" * 81,
            "sk-proj-secret",
        ]:
            for kind in [
                "insufficient_quota",
                "rate_limit_error",
                "credit_balance_exhausted",
            ]:
                with self.subTest(code=code, kind=kind):
                    event = recorded(
                        sdk_error(
                            code,
                            kind,
                            headers={
                                "retry-after": "30",
                                "x-ratelimit-remaining-tokens": "0",
                            },
                        )
                    )
                    self.assertEqual("UNKNOWN_429", event["classification"])
        event = recorded(sdk_error("future_provider_code"))
        self.assertEqual("future_provider_code", event["providerCode"])

    def test_real_sdk_coercion_of_bad_code_is_not_logged_as_a_safe_raw_code(self):
        error = sdk_error(body={"code": {"nested": "secret"}, "type": ["bad"]})
        self.assertIsInstance(error.code, str)
        event = recorded(error)
        self.assertIsNone(event["providerCode"])
        self.assertIsNone(event["providerType"])
        for body in ["private message", [], 42]:
            self.assertEqual(
                "UNKNOWN_429", recorded(sdk_error(body=body))["classification"]
            )

    def test_allowlisted_headers_are_bounded_and_do_not_change_classification(self):
        values = {
            "retry-after": "30",
            "retry-after-ms": "500",
            "x-ratelimit-limit-requests": "600",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-requests": "1m2.5s",
            "x-ratelimit-reset-tokens": "500ms",
        }
        event = recorded(
            sdk_error(
                "new_code",
                headers=values
                | {
                    "authorization": "Bearer PRIVATE_TOKEN",
                    "set-cookie": "PRIVATE_COOKIE",
                    "x-request-id": "req_fixture",
                },
            )
        )
        self.assertEqual(values, event["headers"])
        self.assertEqual("UNKNOWN_429", event["classification"])
        self.assertLess(len(json.dumps(event)), 1500)
        serialized = json.dumps(event)
        for private in [
            "PRIVATE_TOKEN",
            "PRIVATE_COOKIE",
            "PRIVATE_BODY",
            "PRIVATE_PROVIDER_BODY",
            "Fixture",
            "api.openai.com",
        ]:
            self.assertNotIn(private, serialized)

    def test_bad_headers_dates_ids_and_correlation_cannot_log_arbitrary_text(self):
        error = sdk_error()
        error.request_id = "req_" + "x" * 100
        error.response.headers = {
            "retry-after": "Mon, 01 Jan 2024 12:00:00 GMT\r\nPRIVATE",
            "retry-after-ms": "9999999999999999",
            "x-ratelimit-limit-tokens": "a@private.test",
            "x-ratelimit-reset-tokens": "private phrase",
            "authorization": "PRIVATE",
        }
        event = recorded(error, "SECRET_NOT_A_UUID")
        self.assertEqual({}, event["headers"])
        self.assertIsNone(event["operationKey"])
        self.assertIsNone(event["providerRequestId"])
        self.assertNotIn("PRIVATE", json.dumps(event))
        valid_date = recorded(
            sdk_error(headers={"retry-after": "Mon, 01 Jan 2024 12:00:00 GMT"})
        )
        self.assertEqual(
            "Mon, 01 Jan 2024 12:00:00 GMT", valid_date["headers"]["retry-after"]
        )

    def test_non429_and_connection_errors_are_not_quota_diagnoses(self):
        for status in [400, 401, 403, 500, 503]:
            event = recorded(sdk_error("credit_balance_exhausted", status=status))
            self.assertEqual("PROVIDER_ERROR", event["classification"])
            self.assertEqual(status, event["httpStatus"])
        error = openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/responses")
        )
        self.assertIsNone(recorded(error)["httpStatus"])

    def test_logger_and_malformed_accessors_are_best_effort_no_second_event(self):
        with patch.object(
            diagnostics.log, "warning", side_effect=OSError("disk failure")
        ) as logger:
            diagnostics.log_provider_error(sdk_error(), KEY)
            logger.assert_called_once()

        class Broken:
            @property
            def body(self):
                raise RuntimeError("private accessor")

        diagnostics.log_provider_error(Broken(), KEY)

    def test_request_id_uuid_shape_from_real_management_metadata_is_supported(self):
        event = recorded(sdk_error(headers={"x-request-id": KEY}))
        self.assertEqual(KEY, event["providerRequestId"])


class ProductNoninterference(unittest.IsolatedAsyncioTestCase):
    def sdk(self, *, error=None, response=None):
        client = MagicMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        client.responses = SimpleNamespace(
            create=AsyncMock(side_effect=error, return_value=response)
        )
        return client

    def settings(self):
        return patch.multiple(
            adapter.config,
            WEBSITE_SEARCH_PROVIDER="openai",
            OPENAI_API_KEY="OFFLINE_FIXTURE_NOT_A_KEY",
            OPENAI_WEB_SEARCH_MODEL="gpt-5.4-nano",
            OPENAI_WEB_SEARCH_REASONING_EFFORT="low",
        )

    async def test_success_same_request_output_counts_settings_and_no_diagnostic(self):
        raw = {
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"sources": None},
                }
            ],
        }
        response = SimpleNamespace(
            model_dump=lambda: raw,
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )
        client = self.sdk(response=response)
        with (
            self.settings(),
            patch("openai.AsyncOpenAI", return_value=client) as factory,
            patch.object(diagnostics.log, "warning") as logger,
        ):
            result = await adapter.search_sources(BODY)
            self.assertEqual(
                {
                    "sources": [],
                    "usage": {"calls": 1, "inputTokens": 3, "outputTokens": 2},
                },
                result,
            )
            factory.assert_called_once_with(
                api_key="OFFLINE_FIXTURE_NOT_A_KEY", timeout=22, max_retries=0
            )
            client.responses.create.assert_awaited_once()
            kwargs = client.responses.create.call_args.kwargs
            self.assertEqual(1, kwargs["max_tool_calls"])
            self.assertEqual("required", kwargs["tool_choice"])
            logger.assert_not_called()

    async def test_api_errors_keep_external_mapping_retry_after_and_original_observer_identity(
        self,
    ):
        cases = [
            (429, "SOURCE_PROVIDER_QUOTA"),
            (401, "SOURCE_PROVIDER_AUTH"),
            (403, "SOURCE_PROVIDER_AUTH"),
            (500, "SOURCE_PROVIDER_ERROR"),
        ]
        for status, expected in cases:
            error = sdk_error(
                status=status,
                headers={"retry-after": "30", "x-request-id": "req_fixture"},
            )
            client = self.sdk(error=error)
            observer = SimpleNamespace(
                http_client=lambda: "FAKE_TRANSPORT", failure=MagicMock()
            )
            with (
                self.subTest(status=status),
                self.settings(),
                patch("openai.AsyncOpenAI", return_value=client),
                patch.object(
                    diagnostics.log,
                    "warning",
                    side_effect=RuntimeError("logger failure"),
                ) as logger,
            ):
                with self.assertRaises(adapter.SourceAdapterError) as caught:
                    await adapter.search_sources(BODY, observation=observer)
                self.assertEqual(expected, caught.exception.code)
                self.assertEqual({}, caught.exception.details)
                self.assertEqual(
                    "30" if status == 429 else None, caught.exception.retry_after
                )
                observer.failure.assert_called_once_with(error)
                logger.assert_called_once()
                client.responses.create.assert_awaited_once()
                client.__aexit__.assert_awaited_once()

    async def test_cancellation_and_non_sdk_exception_identity_not_logged_or_retried(
        self,
    ):
        for error in [asyncio.CancelledError(), RuntimeError("original")]:
            client = self.sdk(error=error)
            with (
                self.settings(),
                patch("openai.AsyncOpenAI", return_value=client),
                patch.object(diagnostics.log, "warning") as logger,
            ):
                with self.assertRaises(type(error)) as caught:
                    await adapter.search_sources(BODY)
                self.assertIs(error, caught.exception)
                client.responses.create.assert_awaited_once()
                logger.assert_not_called()

    async def test_coalesced_callers_get_original_http_contract_one_diagnostic_one_sdk_call(
        self,
    ):
        source = adapter.EmailSourceAdapter(SimpleNamespace(active_count=0))
        application = web.Application()

        async def search(request):
            return await source.handle(request, "search")

        application.router.add_post("/search", search)
        client = TestClient(TestServer(application))
        await client.start_server()
        sdk = self.sdk(
            error=sdk_error(
                "credit_balance_exhausted",
                "insufficient_quota",
                headers={"retry-after": "30", "x-request-id": "req_fixture"},
            )
        )
        try:
            with (
                self.settings(),
                patch("openai.AsyncOpenAI", return_value=sdk),
                patch.object(diagnostics.log, "warning") as logger,
            ):
                first, second = await asyncio.gather(
                    client.post("/search", json=BODY), client.post("/search", json=BODY)
                )
                for result in [first, second]:
                    self.assertEqual(503, result.status)
                    self.assertEqual(
                        {"error": {"code": "SOURCE_PROVIDER_QUOTA"}},
                        await result.json(),
                    )
                    self.assertEqual("30", result.headers["Retry-After"])
                sdk.responses.create.assert_awaited_once()
                logger.assert_called_once()
                self.assertEqual(
                    KEY, json.loads(logger.call_args.args[1])["operationKey"]
                )
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()

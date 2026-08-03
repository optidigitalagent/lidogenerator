# -*- coding: utf-8 -*-
"""Mocked tests for the disabled-by-default OpenAI lead scorer."""

import asyncio
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from agents import ai_scorer
from models import Business


REPO_ROOT = Path(__file__).resolve().parent.parent
OPENAI_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_SCORING_ENABLED",
    "OPENAI_SCORING_REASONING_EFFORT",
    "OPENAI_SCORING_MAX_OUTPUT_TOKENS",
    "OPENAI_SCORING_TIMEOUT_SECONDS",
)


def _run_isolated_config(overrides=None):
    env = os.environ.copy()
    for name in OPENAI_ENV_NAMES:
        env.pop(name, None)
    env.update(overrides or {})
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = """
import json
import sys
import types

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules["dotenv"] = dotenv

import config
print(json.dumps({
    "key": config.OPENAI_API_KEY,
    "model": config.OPENAI_MODEL,
    "enabled": config.OPENAI_SCORING_ENABLED,
    "reasoning": config.OPENAI_SCORING_REASONING_EFFORT,
    "tokens": config.OPENAI_SCORING_MAX_OUTPUT_TOKENS,
    "timeout": config.OPENAI_SCORING_TIMEOUT_SECONDS,
}))
"""
    with tempfile.TemporaryDirectory(prefix="lidogenerator-openai-config-") as temp_dir:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


class ConfigTests(unittest.TestCase):
    def test_safe_defaults_do_not_read_developer_dotenv(self):
        result = _run_isolated_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "key": "",
                "model": "gpt-5-nano",
                "enabled": "false",
                "reasoning": "minimal",
                "tokens": 512,
                "timeout": 20.0,
            },
        )

    def test_enabled_boolean_is_trimmed_and_casefolded(self):
        result = _run_isolated_config({"OPENAI_SCORING_ENABLED": "  TrUe  "})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["enabled"], "true")

    def test_invalid_boolean_is_rejected(self):
        for value in ("1", "yes", "", "enabled"):
            with self.subTest(value=value):
                result = _run_isolated_config({"OPENAI_SCORING_ENABLED": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("OPENAI_SCORING_ENABLED", result.stderr)

    def test_model_override_is_trimmed(self):
        result = _run_isolated_config({"OPENAI_MODEL": "  test-model  "})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"], "test-model")

    def test_reasoning_effort_allowed_values_are_normalized(self):
        for value in ("minimal", " low ", "MEDIUM", "High"):
            with self.subTest(value=value):
                result = _run_isolated_config(
                    {"OPENAI_SCORING_REASONING_EFFORT": value}
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    json.loads(result.stdout)["reasoning"],
                    value.strip().casefold(),
                )

    def test_invalid_reasoning_effort_is_rejected(self):
        for value in ("none", "", "xhigh", "arbitrary"):
            with self.subTest(value=value):
                result = _run_isolated_config(
                    {"OPENAI_SCORING_REASONING_EFFORT": value}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("OPENAI_SCORING_REASONING_EFFORT", result.stderr)

    def test_output_token_bounds_and_strict_integer(self):
        for value in ("64", "1024"):
            with self.subTest(valid=value):
                result = _run_isolated_config(
                    {"OPENAI_SCORING_MAX_OUTPUT_TOKENS": value}
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("63", "1025", "1.5", "true"):
            with self.subTest(invalid=value):
                result = _run_isolated_config(
                    {"OPENAI_SCORING_MAX_OUTPUT_TOKENS": value}
                )
                self.assertNotEqual(result.returncode, 0)

    def test_timeout_bounds_and_finiteness(self):
        for value in ("0.01", "60"):
            with self.subTest(valid=value):
                result = _run_isolated_config(
                    {"OPENAI_SCORING_TIMEOUT_SECONDS": value}
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("0", "60.1", "nan", "inf", "true"):
            with self.subTest(invalid=value):
                result = _run_isolated_config(
                    {"OPENAI_SCORING_TIMEOUT_SECONDS": value}
                )
                self.assertNotEqual(result.returncode, 0)


class _FakeResponses:
    def __init__(self, queue, calls):
        self._queue = queue
        self._calls = calls

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str) or item is None:
            return types.SimpleNamespace(
                status="completed",
                incomplete_details=None,
                output=[],
                output_text=item,
            )
        return item


def _response(
    *,
    status="completed",
    output_text=None,
    incomplete_reason=None,
    output=None,
    **private_fields,
):
    incomplete_details = (
        types.SimpleNamespace(reason=incomplete_reason)
        if incomplete_reason is not None
        else None
    )
    return types.SimpleNamespace(
        status=status,
        incomplete_details=incomplete_details,
        output=output or [],
        output_text=output_text,
        **private_fields,
    )


def _fake_openai_module(queue):
    module = types.ModuleType("openai")
    request_calls = []
    client_calls = []

    def async_openai(**kwargs):
        client_calls.append(kwargs)
        return types.SimpleNamespace(
            responses=_FakeResponses(queue, request_calls),
        )

    module.AsyncOpenAI = async_openai
    return module, client_calls, request_calls


def _sdk_error(name, *, status_code=None, code=None, param=None):
    class SafeFakeError(Exception):
        def __str__(self):
            raise AssertionError("error text must not be read")

    SafeFakeError.__name__ = name
    error = SafeFakeError("SECRET_RAW_MESSAGE sk-FAKE_API_KEY")
    if status_code is not None:
        error.status_code = status_code
    if code is not None:
        error.code = code
    if param is not None:
        error.param = param
    error.request = types.SimpleNamespace(url="https://api.openai.test/SECRET_URL")
    error.response = types.SimpleNamespace(
        body={"secret": "SECRET_RESPONSE_BODY"},
        headers={"x-request-id": "SECRET_REQUEST_ID"},
    )
    error.request_id = "SECRET_REQUEST_ID"
    return error


class OpenAIScorerTests(unittest.IsolatedAsyncioTestCase):
    def _config_patch(self, *, enabled="true", key="test-key"):
        return patch.multiple(
            ai_scorer.config,
            OPENAI_SCORING_ENABLED=enabled,
            OPENAI_API_KEY=key,
            OPENAI_MODEL="gpt-5-nano",
            OPENAI_SCORING_REASONING_EFFORT="minimal",
            OPENAI_SCORING_MAX_OUTPUT_TOKENS=512,
            OPENAI_SCORING_TIMEOUT_SECONDS=20.0,
        )

    async def test_disabled_uses_no_import_or_client_and_reports_progress(self):
        module, client_calls, request_calls = _fake_openai_module([])
        businesses = [Business(name="One"), Business(name="Two")]
        original_objects = list(businesses)
        progress = []

        async def on_progress(done, total):
            progress.append((done, total))

        with self._config_patch(enabled="false", key=""), patch.dict(
            sys.modules, {"openai": module}
        ):
            result = await ai_scorer.score_businesses(businesses, on_progress)

        self.assertIs(result, businesses)
        self.assertEqual(result, original_objects)
        self.assertEqual(client_calls, [])
        self.assertEqual(request_calls, [])
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertTrue(all(item.ai_score == 50 for item in businesses))
        self.assertTrue(all(item.ai_priority == "warm" for item in businesses))
        self.assertTrue(all(item.ai_reason == "AI-скоринг отключён" for item in businesses))

    async def test_enabled_without_key_uses_safe_fallback_and_zero_requests(self):
        module, client_calls, request_calls = _fake_openai_module([])
        business = Business(name="One")
        with self._config_patch(enabled="true", key=""), patch.dict(
            sys.modules, {"openai": module}
        ):
            await ai_scorer.score_businesses([business])
        self.assertEqual(client_calls, [])
        self.assertEqual(request_calls, [])
        self.assertEqual(
            business.ai_reason,
            "AI-скоринг отключён (нет OPENAI_API_KEY)",
        )

    async def test_success_uses_responses_api_schema_and_private_prompt_boundary(self):
        payload = json.dumps(
            {"score": 82, "priority": "hot", "reason": "Strong opportunity"}
        )
        module, client_calls, request_calls = _fake_openai_module([payload])
        business = Business(
            id=991,
            name="Safe Dental Studio",
            niche="dentistry",
            city="Zaporizhzhia",
            phone="SECRET_PHONE_123",
            address="SECRET_EXACT_ADDRESS",
            instagram_url="https://instagram.com/private_handle",
            instagram_active=True,
            followers=720,
            last_post_days=4,
            reviews_count=31,
            site_quality="bad",
            has_site=True,
            website_resolution_status="found_official",
            website_resolution_evidence="SECRET_RESOLUTION_EVIDENCE",
            website_audit_status="bad",
            website_audit_evidence="SECRET_AUDIT_EVIDENCE",
        )

        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            result = await ai_scorer.score_businesses([business])

        self.assertIs(result[0], business)
        self.assertEqual((business.ai_score, business.ai_priority), (82, "hot"))
        self.assertEqual(business.ai_reason, "Strong opportunity")
        self.assertEqual(client_calls, [{"api_key": "test-key", "timeout": 20.0}])
        self.assertEqual(len(request_calls), 1)
        request = request_calls[0]
        self.assertEqual(request["model"], "gpt-5-nano")
        self.assertEqual(request["max_output_tokens"], 512)
        self.assertEqual(request["reasoning"], {"effort": "minimal"})
        self.assertNotIn("summary", request["reasoning"])
        self.assertEqual(request["tools"], [])
        self.assertNotIn("web_search", request)
        self.assertFalse(request["store"])
        self.assertEqual(request["text"], ai_scorer.SCORING_TEXT_FORMAT)
        self.assertEqual(
            request["text"]["format"]["schema"],
            ai_scorer.SCORING_SCHEMA,
        )
        self.assertTrue(request["text"]["format"]["strict"])
        prompt = request["input"]
        for expected in (
            "Safe Dental Studio",
            "dentistry",
            "Zaporizhzhia",
            "bad website",
            "found_official",
            "Website audit status: bad",
            "Instagram active: active",
            "Instagram followers: 720",
            "Instagram last post days: 4",
            "Maps reviews: 31",
        ):
            self.assertIn(expected, prompt)
        for private_value in (
            "SECRET_PHONE_123",
            "SECRET_EXACT_ADDRESS",
            "SECRET_RESOLUTION_EVIDENCE",
            "SECRET_AUDIT_EVIDENCE",
            "private_handle",
            "991",
        ):
            self.assertNotIn(private_value, prompt)

    async def test_configured_reasoning_effort_is_passed(self):
        payload = json.dumps(
            {"score": 55, "priority": "warm", "reason": "Qualified"}
        )
        module, _, request_calls = _fake_openai_module([payload])
        with self._config_patch(), patch.object(
            ai_scorer.config,
            "OPENAI_SCORING_REASONING_EFFORT",
            "low",
        ), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([Business()])
        self.assertEqual(request_calls[0]["reasoning"], {"effort": "low"})

    async def test_incomplete_output_limits_use_safe_fallback(self):
        for incomplete_reason in ("max_output_tokens", "max_tokens"):
            with self.subTest(incomplete_reason=incomplete_reason):
                response = _response(
                    status="incomplete",
                    incomplete_reason=f"{incomplete_reason}: SECRET_DETAIL",
                    id="SECRET_RESPONSE_ID",
                    usage={"SECRET_USAGE": 42},
                )
                module, _, _ = _fake_openai_module([response])
                business = Business()
                with self._config_patch(), patch.dict(
                    sys.modules, {"openai": module}
                ):
                    await ai_scorer.score_businesses([business])
                self.assertEqual(
                    business.ai_reason,
                    "AI-скоринг недоступен: output_limit",
                )
                for private_value in (
                    "SECRET_DETAIL",
                    "SECRET_RESPONSE_ID",
                    "SECRET_USAGE",
                ):
                    self.assertNotIn(private_value, business.ai_reason)

    async def test_other_incomplete_response_uses_safe_fallback(self):
        response = _response(
            status="incomplete",
            incomplete_reason="content_filter: SECRET_DETAIL",
        )
        module, _, _ = _fake_openai_module([response])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(
            business.ai_reason,
            "AI-скоринг недоступен: incomplete_response",
        )
        self.assertNotIn("SECRET_DETAIL", business.ai_reason)

    async def test_failed_response_uses_safe_fallback(self):
        response = _response(
            status="failed",
            output_text="SECRET_RAW_OUTPUT",
            id="SECRET_RESPONSE_ID",
        )
        module, _, _ = _fake_openai_module([response])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(
            business.ai_reason,
            "AI-скоринг недоступен: response_failed",
        )
        self.assertNotIn("SECRET", business.ai_reason)

    async def test_completed_empty_response_has_distinct_fallback(self):
        module, _, _ = _fake_openai_module(["   "])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(business.ai_reason, "AI вернул пустой ответ")

    async def test_refusal_uses_safe_fallback_without_refusal_text(self):
        response = _response(
            status="completed",
            output=[
                types.SimpleNamespace(
                    type="message",
                    content=[
                        types.SimpleNamespace(
                            type="refusal",
                            refusal="SECRET_REFUSAL_TEXT",
                        )
                    ],
                )
            ],
            id="SECRET_RESPONSE_ID",
            usage={"SECRET_USAGE": 42},
        )
        module, _, _ = _fake_openai_module([response])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(business.ai_reason, "AI-скоринг недоступен: refusal")
        self.assertNotIn("SECRET", business.ai_reason)

    async def test_invalid_structured_outputs_fall_back(self):
        invalid_payloads = (
            "prefix {\"score\": 80, \"priority\": \"hot\", \"reason\": \"x\"}",
            json.dumps([{"score": 80, "priority": "hot", "reason": "x"}]),
            json.dumps({"score": True, "priority": "hot", "reason": "x"}),
            json.dumps({"score": 50.0, "priority": "warm", "reason": "x"}),
            json.dumps({"score": -1, "priority": "cold", "reason": "x"}),
            json.dumps({"score": 101, "priority": "hot", "reason": "x"}),
            json.dumps({"score": 50, "priority": "urgent", "reason": "x"}),
            json.dumps({"score": 50, "priority": "warm", "reason": 7}),
            json.dumps(
                {"score": 50, "priority": "warm", "reason": "x", "extra": 1}
            ),
        )
        module, _, request_calls = _fake_openai_module(list(invalid_payloads))
        businesses = [Business(name=str(index)) for index in range(len(invalid_payloads))]
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses(businesses)
        self.assertEqual(len(request_calls), len(invalid_payloads))
        for business in businesses:
            self.assertEqual(business.ai_score, 50)
            self.assertEqual(business.ai_priority, "warm")
            self.assertEqual(business.ai_reason, ai_scorer.INVALID_RESPONSE_REASON)

    async def test_priority_is_reconciled_and_reason_is_truncated(self):
        payload = json.dumps(
            {"score": 75, "priority": "cold", "reason": "r" * 600}
        )
        module, _, _ = _fake_openai_module([payload])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(business.ai_priority, "hot")
        self.assertEqual(len(business.ai_reason), 500)

    async def test_error_categories_are_safe(self):
        cases = (
            (type("APITimeoutError", (Exception,), {})("SECRET_TIMEOUT"), "timeout"),
            (type("RateLimitError", (Exception,), {})("SECRET_RATE"), "rate_limit"),
            (
                type("AuthenticationError", (Exception,), {})("SECRET_AUTH"),
                "authentication",
            ),
            (type("APIError", (Exception,), {})("SECRET_GENERIC"), "api_error"),
        )
        for error, category in cases:
            with self.subTest(category=category):
                module, _, _ = _fake_openai_module([error])
                business = Business()
                with self._config_patch(), patch.dict(
                    sys.modules, {"openai": module}
                ):
                    await ai_scorer.score_businesses([business])
                self.assertEqual(
                    business.ai_reason,
                    f"AI-скоринг недоступен: {category}",
                )
                self.assertNotIn("SECRET", business.ai_reason)

    def test_complete_error_category_mapping_uses_safe_metadata(self):
        cases = (
            (_sdk_error("APITimeoutError"), "timeout"),
            (TimeoutError("SECRET_TIMEOUT"), "timeout"),
            (
                _sdk_error("RateLimitError", code="rate_limit_exceeded"),
                "rate_limit",
            ),
            (
                _sdk_error(
                    "RateLimitError",
                    status_code=429,
                    code="insufficient_quota",
                ),
                "insufficient_quota",
            ),
            (_sdk_error("AuthenticationError"), "authentication"),
            (_sdk_error("APIStatusError", status_code=401), "authentication"),
            (_sdk_error("APIError", code="invalid_api_key"), "authentication"),
            (_sdk_error("PermissionDeniedError"), "permission"),
            (_sdk_error("APIStatusError", status_code=403), "permission"),
            (_sdk_error("NotFoundError"), "model_not_found"),
            (_sdk_error("APIStatusError", status_code=404), "model_not_found"),
            (_sdk_error("APIError", code="model_not_found"), "model_not_found"),
            (_sdk_error("BadRequestError"), "bad_request"),
            (_sdk_error("APIStatusError", status_code=400), "bad_request"),
            (_sdk_error("APIStatusError", status_code=422), "bad_request"),
            (_sdk_error("APIConnectionError"), "connection"),
            (_sdk_error("InternalServerError"), "server_error"),
            (_sdk_error("APIStatusError", status_code=500), "server_error"),
            (_sdk_error("APIError"), "api_error"),
        )
        for error, category in cases:
            with self.subTest(category=category):
                self.assertEqual(ai_scorer._safe_error_category(error), category)

    def test_error_categories_ignore_unsafe_metadata(self):
        error = _sdk_error(
            "APIError",
            code="arbitrary_secret_code",
            param="arbitrary_secret_param",
        )
        reason = ai_scorer._safe_error_reason(error)
        self.assertEqual(reason, f"{ai_scorer.SCORING_UNAVAILABLE_PREFIX}api_error")
        for private_value in (
            "SECRET_RAW_MESSAGE",
            "sk-FAKE_API_KEY",
            "SECRET_URL",
            "SECRET_RESPONSE_BODY",
            "SECRET_REQUEST_ID",
            "arbitrary_secret_code",
            "arbitrary_secret_param",
        ):
            self.assertNotIn(private_value, reason)

    async def test_bad_request_pipeline_fallback_is_safe(self):
        error = _sdk_error("BadRequestError", status_code=400)
        module, _, _ = _fake_openai_module([error])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(
            business.ai_reason,
            f"{ai_scorer.SCORING_UNAVAILABLE_PREFIX}bad_request",
        )

    async def test_insufficient_quota_pipeline_fallback_is_safe(self):
        error = _sdk_error(
            "RateLimitError",
            status_code=429,
            code="insufficient_quota",
        )
        module, _, _ = _fake_openai_module([error])
        business = Business()
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses([business])
        self.assertEqual(
            business.ai_reason,
            f"{ai_scorer.SCORING_UNAVAILABLE_PREFIX}insufficient_quota",
        )

    async def test_one_business_failure_does_not_stop_the_next(self):
        error = type("APIError", (Exception,), {})("private request details")
        success = json.dumps(
            {"score": 39, "priority": "cold", "reason": "Second succeeded"}
        )
        module, _, request_calls = _fake_openai_module([error, success])
        businesses = [Business(name="First"), Business(name="Second")]
        progress = []

        async def on_progress(done, total):
            progress.append((done, total))

        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            result = await ai_scorer.score_businesses(businesses, on_progress)
        self.assertIs(result, businesses)
        self.assertEqual(len(request_calls), 2)
        self.assertEqual(businesses[0].ai_score, 50)
        self.assertEqual(businesses[1].ai_score, 39)
        self.assertEqual(businesses[1].ai_priority, "cold")
        self.assertEqual(progress, [(1, 2), (2, 2)])

    async def test_one_incomplete_response_does_not_stop_the_next(self):
        incomplete = _response(
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )
        success = json.dumps(
            {"score": 39, "priority": "cold", "reason": "Second succeeded"}
        )
        module, _, request_calls = _fake_openai_module([incomplete, success])
        businesses = [Business(name="First"), Business(name="Second")]
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses(businesses)
        self.assertEqual(len(request_calls), 2)
        self.assertEqual(
            businesses[0].ai_reason,
            "AI-скоринг недоступен: output_limit",
        )
        self.assertEqual(businesses[1].ai_score, 39)
        self.assertEqual(businesses[1].ai_priority, "cold")

    async def test_sdk_import_or_client_failure_is_fail_open(self):
        module = types.ModuleType("openai")

        def fail_client(**kwargs):
            raise ImportError("private sdk failure")

        module.AsyncOpenAI = fail_client
        businesses = [Business(), Business()]
        with self._config_patch(), patch.dict(sys.modules, {"openai": module}):
            await ai_scorer.score_businesses(businesses)
        self.assertTrue(
            all(item.ai_reason == "AI-скоринг недоступен: sdk" for item in businesses)
        )

    def test_priority_boundaries_and_uncertainty_safe_site_status(self):
        self.assertEqual(ai_scorer._priority_from_score(0), "cold")
        self.assertEqual(ai_scorer._priority_from_score(39), "cold")
        self.assertEqual(ai_scorer._priority_from_score(40), "warm")
        self.assertEqual(ai_scorer._priority_from_score(69), "warm")
        self.assertEqual(ai_scorer._priority_from_score(70), "hot")
        self.assertEqual(ai_scorer._priority_from_score(100), "hot")
        uncertain = Business(site_quality="technical_error")
        self.assertIn("uncertain", ai_scorer._site_status(uncertain))
        self.assertIn("do not assume", ai_scorer._site_status(uncertain))

    def test_schema_is_exact_and_source_has_no_legacy_provider_calls(self):
        self.assertEqual(ai_scorer.SCORING_SCHEMA["required"], ["score", "priority", "reason"])
        self.assertFalse(ai_scorer.SCORING_SCHEMA["additionalProperties"])
        self.assertEqual(
            ai_scorer.SCORING_SCHEMA["properties"]["score"],
            {"type": "integer"},
        )
        self.assertEqual(
            ai_scorer.SCORING_SCHEMA["properties"]["priority"]["enum"],
            ["hot", "warm", "cold"],
        )
        forbidden_keywords = {
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "pattern",
            "format",
            "examples",
            "default",
        }

        def schema_keys(value):
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    yield key
                    yield from schema_keys(nested_value)
            elif isinstance(value, list):
                for nested_value in value:
                    yield from schema_keys(nested_value)

        self.assertTrue(
            forbidden_keywords.isdisjoint(schema_keys(ai_scorer.SCORING_SCHEMA))
        )
        source = inspect.getsource(ai_scorer).casefold()
        self.assertNotIn("anthropic", source)
        self.assertNotIn("asyncanthropic", source)
        self.assertNotIn("messages.create", source)
        self.assertNotIn("print(", source)
        self.assertNotIn("str(exc", source)
        self.assertIn("responses.create", source)


if __name__ == "__main__":
    unittest.main()

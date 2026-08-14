"""OpenAI Responses Web Search adapter for official-site discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import unicodedata

from website_candidate_matching import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    SearchProvider,
    SearchProviderError,
    SearchIdentityEvidence,
    SearchRequest,
    SearchResult,
)
from website_resolution import normalize_candidate_url


_INPUT_LIMIT = 2000
_TITLE_LIMIT = 300
_SNIPPET_LIMIT = 1000
_MAX_SAFE_OUTPUT_TOKENS = 4096
_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
_CONTEXT_SIZES = frozenset({"low", "medium", "high"})

OPENAI_WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "name_matches": {"type": "boolean"},
                    "city_matches": {"type": "boolean"},
                    "address_matches": {"type": "boolean"},
                    "phone_matches": {"type": "boolean"},
                    "different_city_detected": {"type": "boolean"},
                },
                "required": [
                    "url",
                    "title",
                    "snippet",
                    "name_matches",
                    "city_matches",
                    "address_matches",
                    "phone_matches",
                    "different_city_detected",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}

OPENAI_WEB_SEARCH_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "official_website_candidates",
        "strict": True,
        "schema": OPENAI_WEB_SEARCH_SCHEMA,
    }
}


def _normalized_choice(value: object, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().casefold()
    if normalized not in allowed:
        raise ValueError(f"{name} has an unsupported value")
    return normalized


@dataclass(frozen=True)
class OpenAIWebSearchSettings:
    api_key: str = field(repr=False)
    model: str = "gpt-5.4-nano"
    reasoning_effort: str = "low"
    search_context_size: str = "low"
    country: str = "UA"
    external_web_access: bool = True
    max_results: int = 5
    max_output_tokens: int = 1024
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        api_key = self.api_key.strip()
        if not api_key:
            raise ValueError("api_key must not be empty")
        object.__setattr__(self, "api_key", api_key)

        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        model = self.model.strip()
        if not model:
            raise ValueError("model must not be empty")
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self,
            "reasoning_effort",
            _normalized_choice(
                self.reasoning_effort,
                "reasoning_effort",
                _REASONING_EFFORTS,
            ),
        )
        object.__setattr__(
            self,
            "search_context_size",
            _normalized_choice(
                self.search_context_size,
                "search_context_size",
                _CONTEXT_SIZES,
            ),
        )

        if not isinstance(self.country, str):
            raise TypeError("country must be a string")
        country = self.country.strip().upper()
        if len(country) != 2 or not country.isascii() or not country.isalpha():
            raise ValueError("country must be two ASCII letters")
        object.__setattr__(self, "country", country)

        if type(self.external_web_access) is not bool:
            raise TypeError("external_web_access must be a bool")
        if type(self.max_results) is not int:
            raise TypeError("max_results must be an integer")
        if not 1 <= self.max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")
        if type(self.max_output_tokens) is not int:
            raise TypeError("max_output_tokens must be an integer")
        if not 256 <= self.max_output_tokens <= 4096:
            raise ValueError("max_output_tokens must be between 256 and 4096")
        if type(self.timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an integer or float")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout):
            raise ValueError("timeout_seconds must be finite")
        if not 0.0 < timeout <= 30.0:
            raise ValueError("timeout_seconds must be greater than 0 and at most 30")
        object.__setattr__(self, "timeout_seconds", timeout)


@dataclass(frozen=True)
class OpenAIWebSearchTelemetry:
    requests_started: int
    requests_succeeded: int
    requests_failed: int
    tool_calls_seen: int
    search_actions_seen: int
    open_page_actions_seen: int
    find_in_page_actions_seen: int
    unknown_actions_seen: int
    sources_seen: int
    identity_candidates_rejected: int
    candidates_returned: int
    tool_call_limit_exceeded: bool
    last_error_category: str | None = None


def _clean_input_value(value: str) -> str:
    cleaned = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in value
    )
    return " ".join(cleaned.split())


def _bounded_value(value: str, limit: int) -> str:
    cleaned = _clean_input_value(value)
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip()


def _suggested_query_variants(
    name: str,
    city: str,
    address: str | None,
    phone: str | None,
) -> tuple[str, ...]:
    if phone is not None and address is not None:
        candidates = (
            f'"{name}" "{phone}"',
            f'"{name}" "{city}" "{address}"',
            f'"{name}" "{city}"',
        )
    elif phone is not None:
        candidates = (
            f'"{name}" "{phone}"',
            f'"{phone}" "{city}"',
            f'"{name}" "{city}"',
        )
    elif address is not None:
        candidates = (
            f'"{name}" "{city}" "{address}"',
            f'"{name}" "{city}"',
            f'"{address}" "{name}"',
        )
    else:
        candidates = (f'"{name}" "{city}"',)
    return tuple(dict.fromkeys(candidates))[:3]


def build_openai_web_search_input(request: SearchRequest) -> str:
    """Build a deterministic privacy-bounded official-site lookup prompt."""

    if not isinstance(request, SearchRequest):
        raise TypeError("request must be a SearchRequest")
    name = _bounded_value(request.business_name, 100)
    city = _bounded_value(request.city, 80)
    address = _bounded_value(request.address, 120) if request.address is not None else None
    phone = _bounded_value(request.phone, 32) if request.phone is not None else None
    instagram = (
        _bounded_value(request.instagram_url, 80)
        if request.instagram_url is not None
        else None
    )
    identity_lines = [
        f"Business name: {name}",
        f"City: {city}",
    ]
    if address is not None:
        identity_lines.append(f"Address: {address}")
    if phone is not None:
        identity_lines.append(f"Phone: {phone}")
    if instagram is not None:
        identity_lines.append(f"Instagram profile: {instagram}")
    identity_lines.append(f"Maximum candidates: {request.max_results}")
    query_variants = tuple(
        f"{index}. {variant}"
        for index, variant in enumerate(
            _suggested_query_variants(name, city, address, phone),
            start=1,
        )
    )
    if address is not None and phone is not None:
        corroboration = (
            "When an address is supplied, exact address evidence is preferred. "
            "Exact phone evidence may substitute for address evidence only when "
            "source evidence clearly associates that phone with the same named "
            "business in the same city. If address and phone corroboration are "
            "both absent or false, return an empty results array. "
        )
    elif address is not None:
        corroboration = (
            "When an address is supplied without a phone, source evidence must "
            "match that address; otherwise return an empty results array. "
        )
    elif phone is not None:
        corroboration = (
            "When a phone is supplied without an address, source evidence must "
            "match that exact phone; otherwise return an empty results array. "
        )
    else:
        corroboration = ""
    instructions = (
        "All identity fields describe one exact business. A same-name business in "
        "any other city is not a candidate. Name and city must both match, and city "
        "must match exactly in meaning. Different city evidence requires rejection. "
        + corroboration
        + "Do not infer ownership from the name alone. Do not return a best guess. "
        "Perform one search action only. Do not use open_page or find_in_page. Put "
        "multiple focused queries into that one search action if needed. Return only "
        "business-owned websites; exclude social media, Maps, directories, "
        "aggregators, marketplaces, booking/review/link-in-bio/news pages. Never "
        "invent a URL. Set every identity boolean only from source evidence. The "
        "deterministic matcher remains final authority.\n\nSuggested query variants:\n"
        + "\n".join(query_variants)
    )
    prompt = "\n".join(identity_lines) + "\n\n" + instructions
    return prompt[:_INPUT_LIMIT].rstrip()


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _contains_refusal(response: object) -> bool:
    output = _field(response, "output")
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        if _field(item, "type") == "refusal":
            return True
        content = _field(item, "content")
        if isinstance(content, (list, tuple)) and any(
            _field(part, "type") == "refusal" for part in content
        ):
            return True
    return False


def _safe_error_attribute(exc: BaseException, name: str) -> object:
    try:
        return getattr(exc, name, None)
    except Exception:
        return None


def _safe_error_code(exc: BaseException) -> str | None:
    code = _safe_error_attribute(exc, "code")
    if not isinstance(code, str) or len(code) > 64:
        return None
    normalized = code.strip().casefold()
    if normalized in {
        "insufficient_quota",
        "model_not_found",
        "invalid_api_key",
        "rate_limit_exceeded",
    }:
        return normalized
    return None


def _safe_error_category(exc: BaseException) -> str:
    error_name = type(exc).__name__.casefold()
    status_code = _safe_error_attribute(exc, "status_code")
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        status_code = None
    code = _safe_error_code(exc)

    if isinstance(exc, TimeoutError) or "timeout" in error_name:
        return "timeout"
    if code == "insufficient_quota":
        return "insufficient_quota"
    if (
        "ratelimit" in error_name
        or "rate_limit" in error_name
        or status_code == 429
        or code == "rate_limit_exceeded"
    ):
        return "rate_limit"
    if (
        "authentication" in error_name
        or status_code == 401
        or code == "invalid_api_key"
    ):
        return "authentication"
    if "permission" in error_name or status_code == 403:
        return "permission"
    if "notfound" in error_name or "not_found" in error_name or status_code == 404 or code == "model_not_found":
        return "model_not_found"
    if "badrequest" in error_name or "bad_request" in error_name or status_code in {400, 422}:
        return "bad_request"
    if "connection" in error_name:
        return "connection"
    if "server" in error_name or (status_code is not None and 500 <= status_code <= 599):
        return "server_error"
    return "api_error"


def _mapped_sdk_error(exc: BaseException) -> tuple[SearchProviderError, str]:
    category = _safe_error_category(exc)
    if category == "timeout":
        return ProviderTimeout("OpenAI web search timed out"), category
    if category == "rate_limit":
        return ProviderRateLimited("OpenAI web search rate limited"), category
    if category in {"authentication", "permission"}:
        return ProviderAuthError("OpenAI web search authentication failed"), category
    if category == "insufficient_quota":
        return ProviderUnavailable("OpenAI web search quota unavailable"), category
    if category == "model_not_found":
        return ProviderUnavailable("OpenAI web search model unavailable"), category
    if category == "bad_request":
        return ProviderUnavailable("OpenAI web search configuration rejected"), category
    if category == "connection":
        return SearchProviderError("OpenAI web search connection failed"), category
    if category == "server_error":
        return SearchProviderError("OpenAI web search service error"), category
    return SearchProviderError("OpenAI web search request failed"), category


_RESULT_KEYS = frozenset({
    "url",
    "title",
    "snippet",
    "name_matches",
    "city_matches",
    "address_matches",
    "phone_matches",
    "different_city_detected",
})
_IDENTITY_BOOLEAN_KEYS = (
    "name_matches",
    "city_matches",
    "address_matches",
    "phone_matches",
    "different_city_detected",
)


def _validated_payload(output_text: object) -> list[dict[str, object]]:
    if not isinstance(output_text, str) or not output_text.strip():
        raise SearchProviderError("OpenAI web search returned empty output")
    try:
        payload = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        raise SearchProviderError("invalid OpenAI web search response") from None
    if not isinstance(payload, dict) or set(payload) != {"results"}:
        raise SearchProviderError("invalid OpenAI web search response")
    items = payload["results"]
    if not isinstance(items, list):
        raise SearchProviderError("invalid OpenAI web search response")
    validated: list[dict[str, object]] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != _RESULT_KEYS
            or not all(isinstance(item[key], str) for key in ("url", "title", "snippet"))
            or not all(type(item[key]) is bool for key in _IDENTITY_BOOLEAN_KEYS)
        ):
            raise SearchProviderError("invalid OpenAI web search response")
        validated.append(item)
    return validated


def _identity_prefilter_allows(item: Mapping[str, object], request: SearchRequest) -> bool:
    if (
        item["name_matches"] is not True
        or item["city_matches"] is not True
        or item["different_city_detected"] is not False
    ):
        return False
    if request.address is not None:
        return item["address_matches"] is True or (
            request.phone is not None and item["phone_matches"] is True
        )
    if request.phone is not None:
        return item["phone_matches"] is True
    return True


class OpenAIWebSearchProvider(SearchProvider):
    """One-call Responses adapter. Mutable telemetry is not concurrency-safe."""

    def __init__(self, settings: OpenAIWebSearchSettings, client: object | None = None) -> None:
        if not isinstance(settings, OpenAIWebSearchSettings):
            raise TypeError("settings must be OpenAIWebSearchSettings")
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
                max_retries=0,
            )
        if not hasattr(client, "responses"):
            raise TypeError("client must expose Responses API resources")
        self._settings = settings
        self._client = client
        self._requests_started = 0
        self._requests_succeeded = 0
        self._requests_failed = 0
        self._tool_calls_seen = 0
        self._search_actions_seen = 0
        self._open_page_actions_seen = 0
        self._find_in_page_actions_seen = 0
        self._unknown_actions_seen = 0
        self._sources_seen = 0
        self._identity_candidates_rejected = 0
        self._candidates_returned = 0
        self._tool_call_limit_exceeded = False
        self._last_error_category: str | None = None

    @property
    def settings(self) -> OpenAIWebSearchSettings:
        return self._settings

    def telemetry(self) -> OpenAIWebSearchTelemetry:
        return OpenAIWebSearchTelemetry(
            requests_started=self._requests_started,
            requests_succeeded=self._requests_succeeded,
            requests_failed=self._requests_failed,
            tool_calls_seen=self._tool_calls_seen,
            search_actions_seen=self._search_actions_seen,
            open_page_actions_seen=self._open_page_actions_seen,
            find_in_page_actions_seen=self._find_in_page_actions_seen,
            unknown_actions_seen=self._unknown_actions_seen,
            sources_seen=self._sources_seen,
            identity_candidates_rejected=self._identity_candidates_rejected,
            candidates_returned=self._candidates_returned,
            tool_call_limit_exceeded=self._tool_call_limit_exceeded,
            last_error_category=self._last_error_category,
        )

    def _process_response(
        self,
        response: object,
        request: SearchRequest,
    ) -> tuple[SearchResult, ...]:
        status = _field(response, "status")
        if status == "incomplete":
            details = _field(response, "incomplete_details")
            reason = _field(details, "reason")
            if isinstance(reason, str) and (
                "max_output_tokens" in reason.casefold()
                or "max_tokens" in reason.casefold()
            ):
                raise SearchProviderError("OpenAI web search output limit reached")
            raise SearchProviderError("OpenAI web search response incomplete")
        if status == "failed":
            raise SearchProviderError("OpenAI web search response failed")
        if _contains_refusal(response):
            raise SearchProviderError("OpenAI web search refused request")
        if status != "completed":
            raise SearchProviderError("OpenAI web search response incomplete")

        output = _field(response, "output")
        output_items = output if isinstance(output, (list, tuple)) else ()
        source_urls: list[str] = []
        source_keys: set[str] = set()
        tool_calls = 0
        sources_seen = 0
        for item in output_items:
            if _field(item, "type") != "web_search_call":
                continue
            tool_calls += 1
            action = _field(item, "action")
            action_type = _field(action, "type")
            if action_type == "search":
                self._search_actions_seen += 1
            elif action_type == "open_page":
                self._open_page_actions_seen += 1
                continue
            elif action_type == "find_in_page":
                self._find_in_page_actions_seen += 1
                continue
            else:
                self._unknown_actions_seen += 1
                continue
            sources = _field(action, "sources")
            if not isinstance(sources, (list, tuple)):
                continue
            for source in sources:
                url = _field(source, "url")
                if not isinstance(url, str):
                    continue
                sources_seen += 1
                try:
                    normalized = normalize_candidate_url(url)
                except (TypeError, ValueError):
                    continue
                if normalized not in source_keys:
                    source_keys.add(normalized)
                    source_urls.append(normalized)
        self._tool_calls_seen += tool_calls
        self._sources_seen += sources_seen
        if tool_calls > 1:
            self._tool_call_limit_exceeded = True
            self._last_error_category = "tool_call_limit"
            raise SearchProviderError("OpenAI web search exceeded tool call limit")
        if tool_calls == 0:
            raise SearchProviderError("OpenAI web search tool was not called")
        items = _validated_payload(_field(response, "output_text"))
        eligible_items: list[dict[str, object]] = []
        for item in items:
            if _identity_prefilter_allows(item, request):
                eligible_items.append(item)
            else:
                self._identity_candidates_rejected += 1
        if eligible_items and not source_urls:
            raise SearchProviderError("OpenAI web search returned unverified candidates")

        count = min(request.max_results, self._settings.max_results)
        results: list[SearchResult] = []
        seen_candidates: set[str] = set()
        for item in eligible_items:
            try:
                normalized_url = normalize_candidate_url(item["url"])
            except (TypeError, ValueError):
                continue
            if normalized_url not in source_keys or normalized_url in seen_candidates:
                continue
            seen_candidates.add(normalized_url)
            results.append(SearchResult(
                normalized_url,
                item["title"][:_TITLE_LIMIT],
                item["snippet"][:_SNIPPET_LIMIT],
                len(results) + 1,
                SearchIdentityEvidence(
                    name_matches=item["name_matches"],
                    city_matches=item["city_matches"],
                    address_matches=item["address_matches"],
                    phone_matches=item["phone_matches"],
                    different_city_detected=item["different_city_detected"],
                    candidate_url_source_bound=True,
                ),
            ))
            if len(results) >= count:
                break
        return tuple(results)

    async def search(self, request: SearchRequest) -> tuple[SearchResult, ...]:
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")
        self._requests_started += 1
        try:
            response = await self._client.responses.create(
                model=self._settings.model,
                reasoning={"effort": self._settings.reasoning_effort},
                tools=[{
                    "type": "web_search",
                    "search_context_size": self._settings.search_context_size,
                    "user_location": {
                        "type": "approximate",
                        "country": self._settings.country,
                        "city": request.city,
                    },
                    "external_web_access": self._settings.external_web_access,
                }],
                tool_choice="required",
                max_tool_calls=1,
                include=["web_search_call.action.sources"],
                input=build_openai_web_search_input(request),
                text=OPENAI_WEB_SEARCH_TEXT_FORMAT,
                max_output_tokens=min(
                    self._settings.max_output_tokens,
                    _MAX_SAFE_OUTPUT_TOKENS,
                ),
                store=False,
            )
        except Exception as exc:
            self._requests_failed += 1
            mapped, category = _mapped_sdk_error(exc)
            self._last_error_category = category
            raise mapped from None

        self._last_error_category = None
        try:
            results = self._process_response(response, request)
        except SearchProviderError:
            self._requests_failed += 1
            if self._last_error_category != "tool_call_limit":
                self._last_error_category = "response_error"
            raise
        self._requests_succeeded += 1
        self._candidates_returned += len(results)
        self._last_error_category = None
        return results

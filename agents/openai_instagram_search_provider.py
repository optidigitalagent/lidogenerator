"""OpenAI Responses Web Search adapter for official Instagram discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import unicodedata

from instagram_candidate_matching import (
    InstagramProviderAuthError,
    InstagramProviderRateLimited,
    InstagramProviderTimeout,
    InstagramProviderUnavailable,
    InstagramSearchIdentityEvidence,
    InstagramSearchProvider,
    InstagramSearchProviderError,
    InstagramSearchRequest,
    InstagramSearchResult,
    normalize_instagram_profile_url,
    normalize_instagram_username,
)
from website_resolution import normalize_domain


_INPUT_LIMIT = 2400
_TITLE_LIMIT = 300
_SNIPPET_LIMIT = 1000
_MAX_SAFE_OUTPUT_TOKENS = 4096
_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
_CONTEXT_SIZES = frozenset({"low", "medium", "high"})

OPENAI_INSTAGRAM_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instagram_url": {"type": "string"},
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "name_matches": {"type": "boolean"},
                    "city_matches": {"type": "boolean"},
                    "address_matches": {"type": "boolean"},
                    "phone_matches": {"type": "boolean"},
                    "website_domain_matches": {"type": "boolean"},
                    "different_city_detected": {"type": "boolean"},
                },
                "required": [
                    "instagram_url", "title", "snippet", "name_matches",
                    "city_matches", "address_matches", "phone_matches",
                    "website_domain_matches", "different_city_detected",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}

OPENAI_INSTAGRAM_SEARCH_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "official_instagram_candidates",
        "strict": True,
        "schema": OPENAI_INSTAGRAM_SEARCH_SCHEMA,
    }
}


def _choice(value: object, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().casefold()
    if normalized not in allowed:
        raise ValueError(f"{name} has an unsupported value")
    return normalized


@dataclass(frozen=True)
class OpenAIInstagramSearchSettings:
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
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        object.__setattr__(self, "api_key", self.api_key.strip())
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(
            self, "reasoning_effort",
            _choice(self.reasoning_effort, "reasoning_effort", _REASONING_EFFORTS),
        )
        object.__setattr__(
            self, "search_context_size",
            _choice(self.search_context_size, "search_context_size", _CONTEXT_SIZES),
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
        if not math.isfinite(timeout) or not 0.0 < timeout <= 30.0:
            raise ValueError("timeout_seconds must be greater than 0 and at most 30")
        object.__setattr__(self, "timeout_seconds", timeout)


@dataclass(frozen=True)
class OpenAIInstagramSearchTelemetry:
    """Aggregate safe counters; no candidate identity or source text is retained.

    ``sources_seen`` counts source URL fields, while ``direct_profile_sources_seen``
    counts the subset that normalize as direct profiles. Structured candidates are
    counted before the identity prefilter. Invalid and source-unbound discard counters
    apply only after that prefilter. The legacy rejection/return counters mirror
    ``identity_prefilter_rejected`` and ``source_bound_candidates_returned``.
    """

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
    structured_candidates_seen: int = 0
    identity_prefilter_rejected: int = 0
    direct_profile_sources_seen: int = 0
    invalid_profile_candidates_discarded: int = 0
    source_unbound_candidates_discarded: int = 0
    source_bound_candidates_returned: int = 0


def _clean(value: str) -> str:
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in value
    )
    return " ".join(without_controls.split())


def _bounded(value: str, limit: int) -> str:
    cleaned = _clean(value)
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip()


def _suggested_query_variants(request: InstagramSearchRequest) -> tuple[str, ...]:
    name = _bounded(request.business_name, 100)
    city = _bounded(request.city, 80)
    address = _bounded(request.address, 120) if request.address is not None else None
    phone = _bounded(request.phone, 32) if request.phone is not None else None
    domain = normalize_domain(request.website_url) if request.website_url is not None else None
    candidates: list[str] = []
    if phone is not None:
        candidates.extend((
            f'"{name}" "{phone}"',
            f'"{name}" "{city}"',
        ))
        if domain is not None:
            candidates.append(f'"{name}" "{domain}"')
        elif address is not None:
            candidates.append(f'"{name}" "{address}"')
    elif domain is not None:
        candidates.extend((
            f'"{name}" "{domain}"',
            f'"{name}" "{city}"',
        ))
        if address is not None:
            candidates.append(f'"{name}" "{address}"')
    elif address is not None:
        candidates.extend((
            f'"{name}" "{city}"',
            f'"{name}" "{address}"',
        ))
    else:
        candidates.append(f'"{name}" "{city}"')
    return tuple(dict.fromkeys(candidates))[:3]


def build_openai_instagram_search_input(request: InstagramSearchRequest) -> str:
    """Build one deterministic, bounded official-profile lookup prompt."""

    if not isinstance(request, InstagramSearchRequest):
        raise TypeError("request must be an InstagramSearchRequest")
    lines = [
        f"Business name: {_bounded(request.business_name, 100)}",
        f"City: {_bounded(request.city, 80)}",
    ]
    if request.address is not None:
        lines.append(f"Address: {_bounded(request.address, 120)}")
    if request.phone is not None:
        lines.append(f"Phone: {_bounded(request.phone, 32)}")
    if request.website_url is not None:
        lines.append(f"Website domain: {normalize_domain(request.website_url)}")
    lines.append(f"Maximum candidates: {request.count}")
    queries = "\n".join(
        f"{index}. {query}"
        for index, query in enumerate(_suggested_query_variants(request), start=1)
    )
    instructions = (
        "Act as a candidate enumerator, not the final official-account decision "
        "maker. Perform exactly one web search action using the identity query "
        "variants below; the tool is restricted to Instagram sources. Do not use "
        "open_page or find_in_page. Return plausible direct Instagram profile "
        "candidates associated with the requested business and city when they are "
        "present in the actual search results. Only return URLs visibly present as "
        "direct instagram.com/<username>/ profile results. Never fabricate an "
        "Instagram URL. Exclude posts, reels, stories, employees or personal "
        "accounts, fan pages, influencers, clearly unrelated same-name businesses, "
        "and clearly wrong-city profiles. Do not omit a plausible direct profile "
        "solely because address, phone, or website-domain evidence is incomplete; "
        "instead set address_matches, phone_matches, and website_domain_matches to "
        "false whenever the corresponding match is not supported. Set name_matches "
        "and city_matches truthfully from the available search evidence. Set "
        "different_city_detected to true when conflicting city evidence is evident. "
        "It is acceptable to return multiple plausible direct profile candidates; "
        "downstream deterministic code performs source binding, identity prefiltering, "
        "matching, ambiguity resolution, and the final official-account decision. "
        "Return an empty results list only when no plausible direct Instagram profile "
        "result is present.\n\nSuggested identity query variants:\n" + queries
    )
    return ("\n".join(lines) + "\n\n" + instructions)[:_INPUT_LIMIT].rstrip()


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


def _error_attribute(exc: BaseException, name: str) -> object:
    try:
        return getattr(exc, name, None)
    except Exception:
        return None


def _error_category(exc: BaseException) -> str:
    name = type(exc).__name__.casefold()
    status = _error_attribute(exc, "status_code")
    if isinstance(status, bool) or not isinstance(status, int):
        status = None
    code = _error_attribute(exc, "code")
    code = code.strip().casefold() if isinstance(code, str) and len(code) <= 64 else None
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return "timeout"
    if code == "insufficient_quota":
        return "insufficient_quota"
    if "ratelimit" in name or "rate_limit" in name or status == 429:
        return "rate_limit"
    if "authentication" in name or status == 401 or code == "invalid_api_key":
        return "authentication"
    if "permission" in name or status == 403:
        return "permission"
    if "notfound" in name or "not_found" in name or status == 404 or code == "model_not_found":
        return "model_not_found"
    if "badrequest" in name or "bad_request" in name or status in {400, 422}:
        return "bad_request"
    if "connection" in name:
        return "connection"
    if "server" in name or (status is not None and 500 <= status <= 599):
        return "server_error"
    return "api_error"


def _mapped_error(exc: BaseException) -> tuple[InstagramSearchProviderError, str]:
    category = _error_category(exc)
    if category == "timeout":
        return InstagramProviderTimeout("OpenAI Instagram search timed out"), category
    if category == "rate_limit":
        return InstagramProviderRateLimited("OpenAI Instagram search rate limited"), category
    if category in {"authentication", "permission"}:
        return InstagramProviderAuthError("OpenAI Instagram search authentication failed"), category
    if category in {"insufficient_quota", "model_not_found", "bad_request"}:
        return InstagramProviderUnavailable("OpenAI Instagram search unavailable"), category
    return InstagramSearchProviderError("OpenAI Instagram search request failed"), category


_RESULT_KEYS = frozenset({
    "instagram_url", "title", "snippet", "name_matches", "city_matches",
    "address_matches", "phone_matches", "website_domain_matches",
    "different_city_detected",
})
_BOOLEAN_KEYS = (
    "name_matches", "city_matches", "address_matches", "phone_matches",
    "website_domain_matches", "different_city_detected",
)


def _validated_payload(output_text: object) -> list[dict[str, object]]:
    if not isinstance(output_text, str) or not output_text.strip():
        raise InstagramSearchProviderError("OpenAI Instagram search returned empty output")
    try:
        payload = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        raise InstagramSearchProviderError("invalid OpenAI Instagram search response") from None
    if not isinstance(payload, dict) or set(payload) != {"results"}:
        raise InstagramSearchProviderError("invalid OpenAI Instagram search response")
    items = payload["results"]
    if not isinstance(items, list):
        raise InstagramSearchProviderError("invalid OpenAI Instagram search response")
    validated: list[dict[str, object]] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != _RESULT_KEYS
            or not all(isinstance(item[key], str) for key in ("instagram_url", "title", "snippet"))
            or not all(type(item[key]) is bool for key in _BOOLEAN_KEYS)
        ):
            raise InstagramSearchProviderError("invalid OpenAI Instagram search response")
        validated.append(item)
    return validated


def _identity_prefilter_allows(
    item: Mapping[str, object], request: InstagramSearchRequest
) -> bool:
    if (
        item["name_matches"] is not True
        or item["city_matches"] is not True
        or item["different_city_detected"] is not False
    ):
        return False
    address = request.address is not None
    phone = request.phone is not None
    website = request.website_url is not None
    if address and phone:
        return (
            item["address_matches"] is True
            or item["phone_matches"] is True
            or (website and item["website_domain_matches"] is True)
        )
    if phone:
        return item["phone_matches"] is True or (
            website and item["website_domain_matches"] is True
        )
    if address:
        return item["address_matches"] is True or (
            website and item["website_domain_matches"] is True
        )
    if website:
        return item["website_domain_matches"] is True
    return True


class OpenAIInstagramSearchProvider(InstagramSearchProvider):
    """Single-call Responses adapter. Mutable telemetry is task-scoped."""

    def __init__(
        self, settings: OpenAIInstagramSearchSettings, client: object | None = None
    ) -> None:
        if not isinstance(settings, OpenAIInstagramSearchSettings):
            raise TypeError("settings must be OpenAIInstagramSearchSettings")
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
        self._structured_candidates_seen = 0
        self._identity_prefilter_rejected = 0
        self._direct_profile_sources_seen = 0
        self._invalid_profile_candidates_discarded = 0
        self._source_unbound_candidates_discarded = 0
        self._source_bound_candidates_returned = 0
        self._tool_call_limit_exceeded = False
        self._last_error_category: str | None = None

    @property
    def settings(self) -> OpenAIInstagramSearchSettings:
        return self._settings

    def telemetry(self) -> OpenAIInstagramSearchTelemetry:
        return OpenAIInstagramSearchTelemetry(
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
            structured_candidates_seen=self._structured_candidates_seen,
            identity_prefilter_rejected=self._identity_prefilter_rejected,
            direct_profile_sources_seen=self._direct_profile_sources_seen,
            invalid_profile_candidates_discarded=(
                self._invalid_profile_candidates_discarded
            ),
            source_unbound_candidates_discarded=(
                self._source_unbound_candidates_discarded
            ),
            source_bound_candidates_returned=self._source_bound_candidates_returned,
        )

    def _process_response(
        self, response: object, request: InstagramSearchRequest
    ) -> tuple[InstagramSearchResult, ...]:
        status = _field(response, "status")
        if status == "failed":
            raise InstagramSearchProviderError("OpenAI Instagram search response failed")
        if status != "completed" or _contains_refusal(response):
            raise InstagramSearchProviderError("OpenAI Instagram search response incomplete")
        output = _field(response, "output")
        output_items = output if isinstance(output, (list, tuple)) else ()
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
                    source_keys.add(normalize_instagram_username(url))
                except (TypeError, ValueError):
                    continue
                self._direct_profile_sources_seen += 1
        self._tool_calls_seen += tool_calls
        self._sources_seen += sources_seen
        if tool_calls > 1:
            self._tool_call_limit_exceeded = True
            self._last_error_category = "tool_call_limit"
            raise InstagramSearchProviderError("OpenAI Instagram search exceeded tool call limit")
        if tool_calls == 0:
            raise InstagramSearchProviderError("OpenAI Instagram search tool was not called")

        payload = _validated_payload(_field(response, "output_text"))
        self._structured_candidates_seen += len(payload)
        eligible: list[dict[str, object]] = []
        for item in payload:
            if _identity_prefilter_allows(item, request):
                eligible.append(item)
            else:
                self._identity_candidates_rejected += 1
                self._identity_prefilter_rejected += 1

        results: list[InstagramSearchResult] = []
        seen: set[str] = set()
        limit = min(request.count, self._settings.max_results)
        for item in eligible:
            try:
                url = normalize_instagram_profile_url(item["instagram_url"])
                identity_key = normalize_instagram_username(url)
            except (TypeError, ValueError):
                self._invalid_profile_candidates_discarded += 1
                continue
            if identity_key not in source_keys:
                self._source_unbound_candidates_discarded += 1
                continue
            if identity_key in seen:
                continue
            seen.add(identity_key)
            results.append(InstagramSearchResult(
                url,
                item["title"][:_TITLE_LIMIT],
                item["snippet"][:_SNIPPET_LIMIT],
                len(results) + 1,
                InstagramSearchIdentityEvidence(
                    name_matches=item["name_matches"],
                    city_matches=item["city_matches"],
                    address_matches=item["address_matches"],
                    phone_matches=item["phone_matches"],
                    website_domain_matches=item["website_domain_matches"],
                    different_city_detected=item["different_city_detected"],
                    candidate_url_source_bound=True,
                ),
            ))
            self._source_bound_candidates_returned += 1
            if len(results) >= limit:
                break
        return tuple(results)

    async def search(
        self, request: InstagramSearchRequest
    ) -> tuple[InstagramSearchResult, ...]:
        if not isinstance(request, InstagramSearchRequest):
            raise TypeError("request must be an InstagramSearchRequest")
        self._requests_started += 1
        try:
            response = await self._client.responses.create(
                model=self._settings.model,
                reasoning={"effort": self._settings.reasoning_effort},
                tools=[{
                    "type": "web_search",
                    "filters": {
                        "allowed_domains": ["instagram.com"],
                    },
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
                input=build_openai_instagram_search_input(request),
                text=OPENAI_INSTAGRAM_SEARCH_TEXT_FORMAT,
                max_output_tokens=min(self._settings.max_output_tokens, _MAX_SAFE_OUTPUT_TOKENS),
                store=False,
            )
        except Exception as exc:
            self._requests_failed += 1
            mapped, category = _mapped_error(exc)
            self._last_error_category = category
            raise mapped from None
        self._last_error_category = None
        try:
            results = self._process_response(response, request)
        except InstagramSearchProviderError:
            self._requests_failed += 1
            if self._last_error_category != "tool_call_limit":
                self._last_error_category = "response_error"
            raise
        self._requests_succeeded += 1
        self._candidates_returned += len(results)
        self._last_error_category = None
        return results

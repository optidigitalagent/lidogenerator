# -*- coding: utf-8 -*-
"""Lead scoring through the OpenAI Responses API.

The scorer uses ``gpt-5-nano`` by default and requests strict structured JSON.
Paid scoring is disabled by default. When it is disabled or unavailable, each
business receives a neutral ``score=50`` / ``priority=warm`` fallback so the
lead pipeline can continue.
"""

import json
from typing import Awaitable, Callable, List, Optional

import config
from models import Business

ProgressCallback = Callable[[int, int], Awaitable[None]]

FALLBACK_SCORE = 50
FALLBACK_PRIORITY = "warm"
INVALID_RESPONSE_REASON = "AI вернул некорректный ответ"

SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "priority": {"type": "string", "enum": ["hot", "warm", "cold"]},
        "reason": {"type": "string"},
    },
    "required": ["score", "priority", "reason"],
    "additionalProperties": False,
}

SCORING_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "lead_score",
        "strict": True,
        "schema": SCORING_SCHEMA,
    }
}

SCORING_INSTRUCTIONS = """You rank already-qualified leads for a Ukrainian web studio.
Return a score from 0 to 100, a matching priority, and a concise reason.
Priority boundaries are: hot >= 70, warm 40-69, cold < 40.
Do not change lead qualification; only rank the supplied lead.

Scoring signals:
- no confirmed website: +35
- old or poor website: +20
- active Instagram (last post under 30 days): +20
- more than 500 followers: +10
- more than 20 Maps reviews: +10
- high-ticket niche such as a clinic, restaurant, or salon: +5

Website resolution or audit uncertainty is not evidence that a website is absent.
Do not award no-website points when the website status is uncertain.
"""


def _site_status(business: Business) -> str:
    """Return an uncertainty-safe website status for the scoring prompt."""
    status = business.website_status
    if status == "uncertain website":
        return "uncertain website (do not assume that no website exists)"
    if status == "no website" and business.site_quality == "dead":
        return "no working website (the supplied site is confirmed dead)"
    return status


def _priority_from_score(score: int) -> str:
    """Derive the only accepted priority from the numeric score."""
    if score >= 70:
        return "hot"
    if score >= 40:
        return "warm"
    return "cold"


def _fallback(business: Business, reason: str) -> None:
    """Apply the pipeline's neutral, fail-open result."""
    business.ai_score = FALLBACK_SCORE
    business.ai_priority = FALLBACK_PRIORITY
    business.ai_reason = reason


def _prompt_for_business(business: Business) -> str:
    """Build the minimal, privacy-bounded business payload for scoring."""
    last_post_days = (
        business.last_post_days
        if business.last_post_days is not None
        else "unknown"
    )
    resolution_status = business.website_resolution_status or "unknown"
    audit_status = business.website_audit_status or "unknown"
    instagram_status = "active" if business.instagram_active else "inactive"
    return (
        f"Business name: {business.name}\n"
        f"Niche: {business.niche}\n"
        f"City: {business.city}\n"
        f"Website status: {_site_status(business)}\n"
        f"Website resolution status: {resolution_status}\n"
        f"Website audit status: {audit_status}\n"
        f"Instagram active: {instagram_status}\n"
        f"Instagram followers: {business.followers}\n"
        f"Instagram last post days: {last_post_days}\n"
        f"Maps reviews: {business.reviews_count}"
    )


def _parse_response(text: object) -> Optional[dict]:
    """Parse one complete JSON object; surrounding or nested free text is rejected."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _validated_result(data: object) -> Optional[tuple[int, str, str]]:
    """Validate the structured result again before it reaches the pipeline."""
    if not isinstance(data, dict) or set(data) != {"score", "priority", "reason"}:
        return None

    score = data["score"]
    priority = data["priority"]
    reason = data["reason"]
    if isinstance(score, bool) or not isinstance(score, int):
        return None
    if not 0 <= score <= 100:
        return None
    if priority not in {"hot", "warm", "cold"}:
        return None
    if not isinstance(reason, str):
        return None

    return score, _priority_from_score(score), reason[:500]


def _safe_error_reason(exc: BaseException) -> str:
    """Map SDK errors without serializing request or response details."""
    error_name = type(exc).__name__.casefold()
    if isinstance(exc, TimeoutError) or "timeout" in error_name:
        category = "timeout"
    elif "ratelimit" in error_name or "rate_limit" in error_name:
        category = "rate_limit"
    elif "authentication" in error_name or "permissiondenied" in error_name:
        category = "authentication"
    else:
        category = "api_error"
    return f"AI-скоринг недоступен: {category}"


async def _score_one(client, business: Business) -> None:
    """Score one business, falling back locally on every API/output failure."""
    try:
        response = await client.responses.create(
            model=config.OPENAI_MODEL,
            instructions=SCORING_INSTRUCTIONS,
            input=_prompt_for_business(business),
            max_output_tokens=config.OPENAI_SCORING_MAX_OUTPUT_TOKENS,
            text=SCORING_TEXT_FORMAT,
            tools=[],
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        validated = _validated_result(_parse_response(output_text))
        if validated is None:
            _fallback(business, INVALID_RESPONSE_REASON)
            return
        business.ai_score, business.ai_priority, business.ai_reason = validated
    except Exception as exc:
        _fallback(business, _safe_error_reason(exc))


async def _fallback_all(
    businesses: List[Business],
    reason: str,
    progress_callback: Optional[ProgressCallback],
) -> List[Business]:
    total = len(businesses)
    for index, business in enumerate(businesses, 1):
        _fallback(business, reason)
        if progress_callback:
            await progress_callback(index, total)
    return businesses


async def score_businesses(
    businesses: List[Business],
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Business]:
    """Score businesses in place and return the original list."""
    if config.OPENAI_SCORING_ENABLED != "true":
        return await _fallback_all(
            businesses,
            "AI-скоринг отключён",
            progress_callback,
        )

    if not config.OPENAI_API_KEY:
        return await _fallback_all(
            businesses,
            "AI-скоринг отключён (нет OPENAI_API_KEY)",
            progress_callback,
        )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_SCORING_TIMEOUT_SECONDS,
        )
    except Exception:
        return await _fallback_all(
            businesses,
            "AI-скоринг недоступен: sdk",
            progress_callback,
        )

    total = len(businesses)
    for index, business in enumerate(businesses, 1):
        await _score_one(client, business)
        if progress_callback:
            await progress_callback(index, total)
    return businesses

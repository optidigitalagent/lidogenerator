# -*- coding: utf-8 -*-
"""Run exactly one explicitly authorized, synthetic OpenAI scoring request."""

import asyncio
import os
from pathlib import Path
import sys


LIVE_GATE = "ALLOW_LIVE_OPENAI_SCORING_VALIDATION"


async def _validate() -> int:
    if os.getenv(LIVE_GATE) != "1":
        print("LIVE_OPENAI_VALIDATION_NOT_RUN_NO_EXPLICIT_OPT_IN")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    try:
        import config
        from agents.ai_scorer import score_businesses
        from models import Business
    except Exception:
        print("LIVE_OPENAI_VALIDATION_NOT_RUN_CONFIGURATION_ERROR")
        return 2

    if config.OPENAI_SCORING_ENABLED != "true":
        print("LIVE_OPENAI_VALIDATION_NOT_RUN_SCORING_DISABLED")
        return 2
    if not config.OPENAI_API_KEY:
        print("LIVE_OPENAI_VALIDATION_NOT_RUN_MISSING_API_KEY")
        return 2
    if not config.OPENAI_MODEL:
        print("LIVE_OPENAI_VALIDATION_NOT_RUN_MISSING_MODEL")
        return 2

    business = Business(
        name="Test Dental Studio",
        niche="стоматологія",
        city="Запоріжжя",
        has_site=False,
        site_quality="none",
        website_resolution_status="not_found",
        website_audit_status="no_official_site",
        instagram_url="https://instagram.com/test_dental_studio_fictional",
        instagram_active=True,
        followers=640,
        last_post_days=5,
        reviews_count=28,
    )
    try:
        await score_businesses([business])
    except Exception:
        print("LIVE_OPENAI_VALIDATION_FAILED")
        return 1
    fallback = business.ai_reason.startswith(("AI-скоринг", "AI вернул"))
    print(f"score={business.ai_score}")
    print(f"priority={business.ai_priority}")
    print(f"reason={business.ai_reason[:200]}")
    print(f"model={config.OPENAI_MODEL}")
    print(f"result={'fallback' if fallback else 'success'}")
    return 1 if fallback else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_validate()))

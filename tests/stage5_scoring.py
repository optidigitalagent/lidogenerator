# -*- coding: utf-8 -*-
"""Stage 5 smoke test: score five saved businesses or use the safe fallback."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config
from agents.ai_scorer import score_businesses
from models import Business


async def main():
    data = json.loads(
        (Path(__file__).parent / "stage3_result.json").read_text(encoding="utf-8")
    )
    items = [Business(**item) for item in data[:5]]

    mode = (
        "OpenAI API"
        if config.OPENAI_SCORING_ENABLED == "true" and config.OPENAI_API_KEY
        else "fallback (scoring disabled)"
    )
    print(f"Scoring mode: {mode}, model: {config.OPENAI_MODEL}\n")

    await score_businesses(items)

    ok = True
    for business in items:
        print(f"- {business.name}")
        print(
            f"  score={business.ai_score}, priority={business.ai_priority}, "
            f"reason={business.ai_reason}"
        )
        if not 0 <= business.ai_score <= 100 or business.ai_priority not in {
            "hot",
            "warm",
            "cold",
        }:
            ok = False

    output_path = Path(__file__).parent / "stage5_result.json"
    output_path.write_text(
        json.dumps([business.to_dict() for business in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTEST PASSED" if ok else "\nTEST FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

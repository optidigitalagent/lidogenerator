# -*- coding: utf-8 -*-
"""Тест этапа 5: AI-скоринг 5 бизнесов из stage3_result.json."""

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
    data = json.loads((Path(__file__).parent / "stage3_result.json").read_text(encoding="utf-8"))
    items = [Business(**d) for d in data[:5]]

    mode = "Claude API" if config.ANTHROPIC_API_KEY else "fallback (нет ключа)"
    print(f"Режим скоринга: {mode}, модель: {config.AI_MODEL}\n")

    await score_businesses(items)

    ok = True
    for b in items:
        print(f"- {b.name}")
        print(f"  score={b.ai_score}, priority={b.ai_priority}, reason={b.ai_reason}")
        if not (0 <= b.ai_score <= 100) or b.ai_priority not in ("hot", "warm", "cold"):
            ok = False

    # Сохраняем результат для следующих этапов
    out = Path(__file__).parent / "stage5_result.json"
    out.write_text(
        json.dumps([b.to_dict() for b in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nТЕСТ ПРОЙДЕН" if ok else "\nТЕСТ ПРОВАЛЕН")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

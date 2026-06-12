# -*- coding: utf-8 -*-
"""Тест Этапа 2: собрать 10 бизнесов "салон краси" в Харькове и вывести в консоль."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents.collector import collect


async def main():
    businesses = await collect("салон краси", "Харків", 10)

    for b in businesses:
        print(f"- {b.name} | {b.phone or 'без тел.'} | {b.address or 'без адреси'}")
        print(f"  сайт: {b.website or '—'} | IG: {b.instagram_url or '—'} | ★{b.rating} ({b.reviews_count} відгуків)")

    print(f"\nВсего собрано: {len(businesses)}")

    # Сохраняем для тестов следующих этапов
    out = Path(__file__).parent / "stage2_result.json"
    out.write_text(
        json.dumps([b.to_dict() for b in businesses], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранено в {out}")

    assert len(businesses) >= 5, "Собрано меньше 5 бизнесов — тест провален"
    print("ТЕСТ ЭТАПА 2: OK")


if __name__ == "__main__":
    asyncio.run(main())

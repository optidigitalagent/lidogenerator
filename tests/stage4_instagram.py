# -*- coding: utf-8 -*-
"""Тест Этапа 4: проверить Instagram для 5 бизнесов из Этапа 2."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents.social_checker import check_instagram
from models import Business


async def main():
    data = json.loads((Path(__file__).parent / "stage3_result.json").read_text(encoding="utf-8"))
    all_biz = [Business(**d) for d in data]

    # Берём 5 бизнесов: сначала те, у кого есть Instagram, потом остальные
    with_ig = [b for b in all_biz if b.instagram_url]
    without_ig = [b for b in all_biz if not b.instagram_url]
    businesses = (with_ig + without_ig)[:5]

    await check_instagram(businesses)

    for b in businesses:
        if b.instagram_url:
            print(f"- {b.name}: IG={b.instagram_url}")
            print(f"  active={b.instagram_active}, followers={b.followers}, "
                  f"posts={b.posts_count}, last_post_days={b.last_post_days}")
        else:
            print(f"- {b.name}: Instagram не вказано")

    # Сохраняем объединённый результат для следующих этапов
    merged = {b.name: b for b in all_biz}
    for b in businesses:
        merged[b.name] = b
    out = Path(__file__).parent / "stage4_result.json"
    out.write_text(
        json.dumps([b.to_dict() for b in merged.values()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    checked = [b for b in businesses if b.instagram_url]
    print(f"\nПроверено профилей: {len(checked)}")
    print("ТЕСТ ЭТАПА 4: OK")


if __name__ == "__main__":
    asyncio.run(main())

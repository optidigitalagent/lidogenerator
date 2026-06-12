# -*- coding: utf-8 -*-
"""Тест Этапа 3: прогнать результаты Этапа 2 через site_checker."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents.site_checker import check_sites
from models import Business


async def main():
    data = json.loads((Path(__file__).parent / "stage2_result.json").read_text(encoding="utf-8"))
    businesses = [Business(**d) for d in data]

    await check_sites(businesses)

    with_site = [b for b in businesses if b.has_site]
    without_site = [b for b in businesses if not b.has_site]

    for b in businesses:
        print(f"- {b.name}: сайт={b.website or '—'} → has_site={b.has_site}, quality={b.site_quality}")

    print(f"\nС сайтом: {len(with_site)} | Без сайта: {len(without_site)}")

    # Сохраняем для следующих этапов
    out = Path(__file__).parent / "stage3_result.json"
    out.write_text(
        json.dumps([b.to_dict() for b in businesses], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert len(with_site) + len(without_site) == len(businesses)
    print("ТЕСТ ЭТАПА 3: OK")


if __name__ == "__main__":
    asyncio.run(main())

# -*- coding: utf-8 -*-
"""Тест Этапа 7: строгая фильтрация лидов + минимальная таблица.

Проверяем главное требование ТЗ:
  - лид = есть Instagram И (нет сайта ИЛИ сайт плохой);
  - бизнес с хорошим сайтом отсеивается;
  - бизнес без Instagram отсеивается;
  - в таблице ровно 4 колонки.

Часть A — детерминированный синтетический датасет на все ветки (без сети).
Часть B — живой прогон site_checker на реальных URL (если есть интернет).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents import reporter, site_checker
from models import Business


def make(name, ig, has_site, quality):
    return Business(
        name=name, niche="барбершоп", city="Харків",
        instagram_url=ig, has_site=has_site, site_quality=quality,
    )


# Синтетика: (имя, instagram, has_site, site_quality) — все возможные комбинации
SYNTH = [
    make("BarberOne (IG, без сайта)",        "https://instagram.com/barber_one", False, "none"),
    make("Fade Club (IG, сайт не открылся)", "https://instagram.com/fade_club",  False, "dead"),
    make("OldCutz (IG, booksy вместо сайта)", "https://instagram.com/oldcutz",   False, "none"),
    make("RetroShave (IG, плохой сайт)",     "https://instagram.com/retroshave", True,  "bad"),
    make("StylePro (IG, хороший сайт)",      "https://instagram.com/stylepro",   True,  "good"),
    make("NoName Barber (без IG, без сайта)", "",                                False, "none"),
    make("Premium Cuts (без IG, хороший сайт)", "",                             True,  "good"),
    make("Lux Barber (без IG, плохой сайт)",  "",                               True,  "bad"),
]


def stats(businesses):
    total = len(businesses)
    leads = [b for b in businesses if b.is_lead]
    no_instagram = sum(1 for b in businesses if not b.instagram_url)
    skipped_good = sum(1 for b in businesses
                       if b.instagram_url and b.website_status == "good website")
    added_no_site = sum(1 for b in leads if b.website_status == "no website")
    added_bad_site = sum(1 for b in leads if b.website_status == "bad website")
    return total, leads, no_instagram, skipped_good, added_no_site, added_bad_site


def part_a():
    print("=" * 60)
    print("ЧАСТЬ A — синтетический датасет (логика фильтра)")
    print("=" * 60)

    total, leads, no_ig, skipped_good, added_no, added_bad = stats(SYNTH)

    print(f"Найдено всего:                {total}")
    print(f"Пропущено (хороший сайт):     {skipped_good}")
    print(f"Пропущено (без Instagram):    {no_ig}")
    print(f"Добавлено без сайта:          {added_no}")
    print(f"Добавлено с плохим сайтом:    {added_bad}")
    print(f"ИТОГО лидов в таблице:        {len(leads)}")
    print()

    print("Примеры итоговых лидов:")
    for b in leads:
        print(f"  - {b.name} | {b.city} | {b.instagram_url} | {b.website_status}")
    print()

    # --- Проверки соответствия ТЗ ---
    names = {b.name for b in leads}
    assert "StylePro (IG, хороший сайт)" not in names, "хороший сайт НЕ должен попадать!"
    assert not any(b for b in leads if not b.instagram_url), "лид без Instagram недопустим!"
    assert all(b.website_status in ("no website", "bad website") for b in leads)
    assert len(leads) == 4, f"ожидалось 4 лида, получено {len(leads)}"
    assert added_no == 3 and added_bad == 1 and skipped_good == 1 and no_ig == 3

    # CSV: ровно 4 колонки, только лиды
    csv_path = reporter.export_csv(SYNTH, task_id=7)
    header = Path(csv_path).read_text(encoding="utf-8-sig").splitlines()[0]
    cols = header.split(";")
    print(f"CSV: {csv_path}")
    print(f"Колонки CSV: {cols}")
    assert cols == ["Business Name", "City", "Instagram URL", "Website Status"], cols
    rows = Path(csv_path).read_text(encoding="utf-8-sig").splitlines()[1:]
    assert len(rows) == 4, f"в CSV должно быть 4 лида, а {len(rows)}"
    print()

    print("Короткий формат для Telegram:")
    print("-" * 40)
    print(reporter.format_leads_summary(leads))
    print("-" * 40)
    print("\n✅ ЧАСТЬ A: все проверки пройдены\n")


async def part_b():
    print("=" * 60)
    print("ЧАСТЬ B — живой прогон site_checker на реальных URL")
    print("=" * 60)
    samples = [
        make("Stripe (современный)",       "https://instagram.com/x", True, "none"),
        make("Booksy-ссылка вместо сайта", "https://instagram.com/y", True, "none"),
    ]
    samples[0].website = "https://stripe.com"
    samples[1].website = "https://booksy.com/somesalon"

    try:
        await asyncio.wait_for(site_checker.check_sites(samples), timeout=40)
    except Exception as e:
        print(f"⚠ Нет сети / прогон не удался ({type(e).__name__}): {e}")
        print("   (это нормально в офлайн-окружении; логика проверена в части A)")
        return

    for b in samples:
        print(f"  - {b.name}: {b.website} → has_site={b.has_site}, "
              f"quality={b.site_quality} → {b.website_status}")
    # booksy — это не сайт: должен быть no website
    assert samples[1].website_status == "no website", "booksy должен считаться 'no website'"
    print("\n✅ ЧАСТЬ B: site_checker отработал\n")


if __name__ == "__main__":
    part_a()
    asyncio.run(part_b())
    print("ТЕСТ ЭТАПА 7: OK")

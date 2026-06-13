# -*- coding: utf-8 -*-
"""Reporter — экспорт лидов в CSV.

В таблицу попадают ТОЛЬКО лиды (есть Instagram И нет сайта/сайт плохой) —
бизнесы с хорошими сайтами и без Instagram отсеиваются.

Колонок ровно четыре, ничего лишнего:
    Business Name | City | Instagram URL | Website Status

Файл пишется в UTF-8 с BOM (Excel сразу показывает кириллицу правильно),
разделитель ";" (Excel с украинской/русской локалью корректно делит колонки).
Сначала идут бизнесы без сайта, затем с плохим сайтом.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import config
from models import Business

# Колонки CSV: (заголовок, как достать значение из Business)
COLUMNS = [
    ("Business Name", lambda b: b.name),
    ("City", lambda b: b.city),
    ("Instagram URL", lambda b: b.instagram_url),
    ("Website Status", lambda b: b.website_status),
]


def export_csv(businesses: List[Business], task_id: int = 0) -> str:
    """Записать CSV только с качественными лидами, вернуть путь к файлу."""
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.EXPORT_DIR / f"leads_task{task_id}_{stamp}.csv"

    # Страховка: даже если на вход пришли все бизнесы — пишем только лидов.
    leads = [b for b in businesses if b.is_lead]
    # Сначала "no website" (главный приоритет), потом "bad website".
    leads.sort(key=lambda b: 0 if b.website_status == "no website" else 1)

    # utf-8-sig = UTF-8 с BOM; newline="" обязателен для модуля csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([header for header, _ in COLUMNS])
        for b in leads:
            writer.writerow([str(getter(b) or "") for _, getter in COLUMNS])

    return str(path)


def format_leads_summary(businesses: List[Business], limit: int = 20) -> str:
    """Короткий список лидов для Telegram. Без скоринга и лишних описаний.

    Формат одного лида:
        Назва: ...
        Місто: ...
        Instagram: ...
        Статус сайту: no website / bad website
    """
    leads = [b for b in businesses if b.is_lead]
    leads.sort(key=lambda b: 0 if b.website_status == "no website" else 1)
    if not leads:
        return "Якісних лідів не знайдено (усі або з нормальним сайтом, або без Instagram)."

    blocks = []
    for b in leads[:limit]:
        blocks.append(
            f"Назва: {b.name}\n"
            f"Місто: {b.city}\n"
            f"Instagram: {b.instagram_url}\n"
            f"Статус сайту: {b.website_status}"
        )
    text = "\n\n".join(blocks)
    if len(leads) > limit:
        text += f"\n\n…та ще {len(leads) - limit} лідів — дивись CSV."
    return text


if __name__ == "__main__":
    # Ручной запуск на тестовых данных
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(
        (Path(__file__).parent.parent / "tests" / "stage5_result.json").read_text(encoding="utf-8")
    )
    items = [Business(**d) for d in data]
    csv_path = export_csv(items, task_id=0)
    print(f"CSV создан: {csv_path}")

# -*- coding: utf-8 -*-
"""Reporter — экспорт результатов в CSV.

Файл пишется в UTF-8 с BOM (Excel сразу показывает кириллицу правильно),
разделитель ";" (Excel с украинской/русской локалью корректно делит колонки),
строки отсортированы по ai_score по убыванию — горячие лиды сверху.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List

import config
from models import Business

# Колонки CSV: (заголовок для человека, имя поля Business)
COLUMNS = [
    ("Назва", "name"),
    ("Ніша", "niche"),
    ("Місто", "city"),
    ("Телефон", "phone"),
    ("Адреса", "address"),
    ("Сайт", "website"),
    ("Instagram", "instagram_url"),
    ("Рейтинг Maps", "rating"),
    ("Відгуків", "reviews_count"),
    ("Є сайт", "has_site"),
    ("Якість сайту", "site_quality"),
    ("Instagram активний", "instagram_active"),
    ("Підписники", "followers"),
    ("Постів", "posts_count"),
    ("Днів з останнього поста", "last_post_days"),
    ("AI-оцінка", "ai_score"),
    ("Пріоритет", "ai_priority"),
    ("Пояснення AI", "ai_reason"),
]


def _format_value(value) -> str:
    """Перевести значение поля в человекочитаемую строку."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "так" if value else "ні"
    return str(value)


def export_csv(businesses: List[Business], task_id: int = 0) -> str:
    """Записать CSV с лидами, вернуть путь к файлу."""
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.EXPORT_DIR / f"leads_task{task_id}_{stamp}.csv"

    # Горячие лиды сверху
    ordered = sorted(businesses, key=lambda b: b.ai_score, reverse=True)

    # utf-8-sig = UTF-8 с BOM; newline="" обязателен для модуля csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([header for header, _ in COLUMNS])
        for b in ordered:
            data = b.to_dict()
            writer.writerow([_format_value(data[field]) for _, field in COLUMNS])

    return str(path)


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

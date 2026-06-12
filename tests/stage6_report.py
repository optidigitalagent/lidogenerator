# -*- coding: utf-8 -*-
"""Тест этапа 6: экспорт CSV и проверка, что колонки читаются корректно."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from agents.reporter import export_csv, COLUMNS
from models import Business


def main():
    data = json.loads((Path(__file__).parent / "stage5_result.json").read_text(encoding="utf-8"))
    items = [Business(**d) for d in data]
    # Делаем разные оценки, чтобы проверить сортировку
    for i, b in enumerate(items):
        b.ai_score = 30 + i * 15

    path = export_csv(items, task_id=999)
    print(f"CSV создан: {path}")

    # Проверка 1: файл начинается с BOM
    raw = Path(path).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "Нет BOM в начале файла"
    print("BOM: есть")

    # Проверка 2: колонки читаются обратно и совпадают с заголовками
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    headers = rows[0]
    assert headers == [h for h, _ in COLUMNS], f"Заголовки не совпадают: {headers}"
    assert len(rows) == len(items) + 1, "Число строк не совпадает"
    assert all(len(r) == len(COLUMNS) for r in rows), "Число колонок в строках не совпадает"
    print(f"Колонки: {len(headers)} шт., строк данных: {len(rows) - 1}")

    # Проверка 3: сортировка по ai_score по убыванию
    scores = [int(r[headers.index("AI-оцінка")]) for r in rows[1:]]
    assert scores == sorted(scores, reverse=True), f"Нет сортировки по убыванию: {scores}"
    print(f"Сортировка по score: {scores}")

    # Проверка 4: кириллица не побилась
    assert any("Харків" in cell or "краси" in cell for r in rows[1:] for cell in r), \
        "Кириллица не найдена в данных"
    print("Кириллица: читается")

    print("\nТЕСТ ПРОЙДЕН")


if __name__ == "__main__":
    main()

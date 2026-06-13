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

    path = export_csv(items, task_id=999)
    print(f"CSV создан: {path}")

    # Сколько из набора реально являются лидами (есть IG и нет/плохой сайт)
    expected_leads = [b for b in items if b.is_lead]

    # Проверка 1: файл начинается с BOM
    raw = Path(path).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "Нет BOM в начале файла"
    print("BOM: есть")

    # Проверка 2: ровно 4 колонки и точные заголовки ТЗ
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    headers = rows[0]
    assert headers == [h for h, _ in COLUMNS], f"Заголовки не совпадают: {headers}"
    assert headers == ["Business Name", "City", "Instagram URL", "Website Status"], headers
    assert all(len(r) == 4 for r in rows), "В каждой строке должно быть 4 колонки"
    print(f"Колонки: {headers}")

    # Проверка 3: в CSV только лиды (хорошие сайты и без IG отсеяны)
    assert len(rows) - 1 == len(expected_leads), \
        f"В CSV {len(rows) - 1} строк, а лидов {len(expected_leads)}"
    print(f"Строк данных (лидов): {len(rows) - 1} из {len(items)} бизнесов")

    # Проверка 4: статус только из разрешённых значений
    statuses = {r[3] for r in rows[1:]}
    assert statuses <= {"no website", "bad website"}, f"Лишние статусы: {statuses}"
    print(f"Статусы сайтов в таблице: {statuses or '∅'}")

    print("\nТЕСТ ПРОЙДЕН")


if __name__ == "__main__":
    main()

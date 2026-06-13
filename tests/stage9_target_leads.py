# -*- coding: utf-8 -*-
"""Тест target_leads: число = сколько ЛИДОВ в таблице, а не сколько просмотреть.

Google Maps и сетевые проверки замоканы, поэтому тест детерминирован и не
ходит в сеть. Проверяем главное новое поведение оркестратора:
  - агент просматривает БОЛЬШЕ бизнесов, чем target;
  - в таблицу попадает РОВНО target валидных лидов;
  - корректно считаются пропуски (без IG / хороший сайт);
  - если лидов меньше target — это честно отражается (shortage).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import db
import orchestrator
from agents import ai_scorer, collector, reporter, site_checker, social_checker
from models import Business


def _make_batch(start_idx: int, size: int) -> list:
    """Батч из size бизнесов. Паттерн на 5: 3 без IG, 1 хороший сайт, 1 валидный лид."""
    batch = []
    for i in range(start_idx, start_idx + size):
        kind = i % 5
        if kind in (0, 1, 2):
            # нет Instagram -> пропуск
            b = Business(name=f"NoIG #{i}", city="Тестмісто", niche="спа")
        elif kind == 3:
            # есть IG, но хороший сайт -> пропуск
            b = Business(name=f"GoodSite #{i}", city="Тестмісто", niche="спа",
                         instagram_url=f"https://instagram.com/good_{i}",
                         has_site=True, site_quality="good")
        else:
            # есть IG, сайта нет -> валидный лид
            b = Business(name=f"Lead #{i}", city="Тестмісто", niche="спа",
                         instagram_url=f"https://instagram.com/lead_{i}",
                         has_site=False, site_quality="none")
        batch.append(b)
    return batch


def _install_fakes(total_available: int, batch_size: int = 10):
    """Подменяем сеть: collect_stream отдаёт синтетику, проверки — no-op."""

    async def fake_collect_stream(niche, city, progress_callback=None, stop_flag=None, **kw):
        visited = 0
        idx = 0
        while visited < total_available:
            if stop_flag and stop_flag():
                return
            n = min(batch_size, total_available - visited)
            batch = _make_batch(idx, n)
            # эмулируем просмотр каждой карточки
            for _ in range(n):
                visited += 1
                if progress_callback:
                    await progress_callback(visited)
            idx += n
            yield batch

    async def fake_check_sites(businesses, progress_callback=None):
        # site_quality/has_site уже проставлены в синтетике — ничего не делаем
        return businesses

    async def fake_check_instagram(businesses, progress_callback=None, stop_flag=None):
        for b in businesses:
            b.instagram_active = True
        return businesses

    async def fake_score(businesses, progress_callback=None):
        for b in businesses:
            b.ai_score, b.ai_priority, b.ai_reason = 50, "warm", "test"
        return businesses

    collector.collect_stream = fake_collect_stream
    site_checker.check_sites = fake_check_sites
    social_checker.check_instagram = fake_check_instagram
    ai_scorer.score_businesses = fake_score


def _leads_in_file(path: str) -> int:
    rows = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return len(rows) - 1  # минус заголовок


async def run_case(target: int, total_available: int) -> dict:
    db.init_db()
    task_id = db.create_task("спа", "Тестмісто", target)
    messages = []

    async def progress(text):
        messages.append(text)

    out_path = await orchestrator.run_search(task_id, progress_callback=progress, progress_interval=0)
    csv_path = next(iter(Path("exports").glob(f"leads_task{task_id}_*.csv")), None)
    # Финальный отчёт — сообщение, которое начинается с "✅ Готово!"
    report = next((m for m in reversed(messages) if m.startswith("✅ Готово!")), "")
    return {
        "task_id": task_id,
        "out_path": out_path,
        "csv_leads": _leads_in_file(str(csv_path)) if csv_path else 0,
        "summary": report,
    }


async def main():
    print("=" * 60)
    print("ТЕСТ 1: target=10, в источнике 50 бизнесов (только 1 из 5 — лид)")
    print("=" * 60)
    _install_fakes(total_available=50, batch_size=10)
    r = await run_case(target=10, total_available=50)
    print(f"Файл: {r['out_path']}")
    print(f"Лидов в CSV: {r['csv_leads']}")
    # В источнике 50 бизнесов, лидов всего 10 (1 из 5). target=10 -> ровно 10.
    assert r["csv_leads"] == 10, f"ожидалось 10 лидов, получено {r['csv_leads']}"
    assert r["out_path"].endswith(".xlsx"), "должен экспортироваться xlsx"
    print("✅ Ровно 10 лидов в таблице, хотя просмотрено 50 бизнесов\n")

    print("=" * 60)
    print("ТЕСТ shortage: target=20, но в источнике всего 50 (=> только 10 лидов)")
    print("=" * 60)
    _install_fakes(total_available=50, batch_size=10)
    r2 = await run_case(target=20, total_available=50)
    print(f"Лидов в CSV: {r2['csv_leads']}")
    print("--- финальный отчёт ---")
    print(r2["summary"])
    print("-----------------------")
    assert r2["csv_leads"] == 10, f"ожидалось 10 лидов (всё, что есть), получено {r2['csv_leads']}"
    assert "знайдено лише 10" in r2["summary"], "должно честно сообщить о нехватке лидов"
    assert "Переглянуто бізнесів: 50" in r2["summary"], "должно показать просмотр всех 50"
    print("✅ Честно: запрошено 20, найдено 10 (источник исчерпан)\n")

    print("ТЕСТ 1: OK")


if __name__ == "__main__":
    asyncio.run(main())

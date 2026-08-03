# -*- coding: utf-8 -*-
"""Конфигурация: читает .env с токенами и настройками."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта — папка, где лежит этот файл
BASE_DIR = Path(__file__).resolve().parent

# Загружаем переменные из .env (если файла нет — используются переменные окружения)
load_dotenv(BASE_DIR / ".env")

# --- Токены ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- База данных ---
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "lead_hunter.db"))

# --- Папка для CSV-экспорта ---
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", str(BASE_DIR / "exports")))

# --- Лимиты парсинга ---
MAX_BUSINESSES = 200            # потолок для устаревшего collect() (по бизнесам)
COLLECT_DELAY_MIN = 3           # задержка между карточками, сек (от)
COLLECT_DELAY_MAX = 6           # задержка между карточками, сек (до)

# --- Логика target_leads (сбор бизнесов батчами до набора нужного числа лидов) ---
# Число, выбранное пользователем, теперь означает СКОЛЬКО ВАЛИДНЫХ ЛИДОВ нужно
# получить в таблице, а НЕ сколько бизнесов просмотреть. Сбор идёт батчами:
# собрали батч -> проверили сайты -> отфильтровали -> добавили лиды -> повторяем,
# пока не наберём target_leads или не упрёмся в safety-лимиты ниже.
COLLECT_BATCH_SIZE = 15          # сколько бизнесов собирать за один батч перед фильтрацией
# Legacy per-stream/default limit for standalone collector.collect_stream() calls.
MAX_BUSINESSES_PER_SEARCH = 1000
# Task-global limit for unique candidates that complete the site checker.
MAX_CHECKED_CANDIDATES_PER_TASK = 1000
# Task-global limit for Maps cards actually opened across all query streams.
MAX_MAPS_CARDS_PER_TASK = 1000
MAX_SCROLL_ROUNDS = 20           # safety: не больше 20 scroll-итераций списка Google Maps
COLLECT_STALE_ROUNDS = 3         # столько scroll-итераций подряд без новых карточек = конец списка
INSTAGRAM_DELAY_MIN = 5         # задержка между профилями Instagram, сек (от)
INSTAGRAM_DELAY_MAX = 8         # задержка между профилями Instagram, сек (до)
SITE_CHECK_TIMEOUT = 10         # таймаут проверки сайта, сек
SITE_CHECK_CONCURRENCY = 5      # не более 5 одновременных HTTP-запросов

WEBSITE_RESOLVER_MODE = os.getenv("WEBSITE_RESOLVER_MODE", "shadow").strip().casefold()
if WEBSITE_RESOLVER_MODE not in {"off", "shadow", "strict"}:
    raise ValueError("WEBSITE_RESOLVER_MODE must be one of: off, shadow, strict")

# --- AI-скоринг ---
# В техплане указан claude-haiku-3-5, но он выведен из эксплуатации (19.02.2026).
# claude-haiku-4-5 — официальная замена, та же цена ($1/$5 за 1M токенов).
AI_MODEL = "claude-haiku-4-5"

# --- Прогресс ---
PROGRESS_INTERVAL = 180         # отправлять прогресс каждые 3 минуты (180 сек)

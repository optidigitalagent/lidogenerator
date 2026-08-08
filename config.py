# -*- coding: utf-8 -*-
"""Конфигурация: читает .env с токенами и настройками."""

import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта — папка, где лежит этот файл
BASE_DIR = Path(__file__).resolve().parent

# Загружаем переменные из .env (если файла нет — используются переменные окружения)
load_dotenv(BASE_DIR / ".env")

# --- Токены ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

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


def _environment_integer(
    name: str,
    default: str,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, default).strip()
    if re.fullmatch(r"[+-]?[0-9]+", raw) is None:
        raise ValueError(f"{name} must be an integer")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _environment_float(
    name: str,
    default: str,
    minimum_exclusive: float,
    maximum: float,
) -> float:
    raw = os.getenv(name, default).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value) or not minimum_exclusive < value <= maximum:
        raise ValueError(
            f"{name} must be greater than {minimum_exclusive:g} and at most {maximum:g}"
        )
    return value


def _environment_boolean(name: str, default: str) -> str:
    raw = os.getenv(name, default).strip().casefold()
    if raw not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return raw


WEBSITE_SEARCH_PROVIDER = os.getenv("WEBSITE_SEARCH_PROVIDER", "none").strip().casefold()
if WEBSITE_SEARCH_PROVIDER not in {"none", "brave", "openai"}:
    raise ValueError("WEBSITE_SEARCH_PROVIDER must be one of: none, brave, openai")

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
BRAVE_SEARCH_COUNTRY = os.getenv("BRAVE_SEARCH_COUNTRY", "UA").strip()
BRAVE_SEARCH_LANGUAGE = os.getenv("BRAVE_SEARCH_LANGUAGE", "").strip()
BRAVE_SEARCH_UI_LANGUAGE = os.getenv("BRAVE_SEARCH_UI_LANGUAGE", "uk-UA").strip()
BRAVE_SEARCH_SAFESEARCH = os.getenv(
    "BRAVE_SEARCH_SAFESEARCH",
    "moderate",
).strip().casefold()
BRAVE_SEARCH_MAX_RESULTS = _environment_integer("BRAVE_SEARCH_MAX_RESULTS", "5", 1, 10)
BRAVE_SEARCH_TIMEOUT_SECONDS = _environment_float(
    "BRAVE_SEARCH_TIMEOUT_SECONDS",
    "10",
    0.0,
    30.0,
)
MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK = _environment_integer(
    "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK",
    "0",
    0,
    1000,
)

# --- AI-скоринг ---
# OpenAI scoring is opt-in so imports and paid requests stay off by default.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_WEB_SEARCH_MODEL = os.getenv(
    "OPENAI_WEB_SEARCH_MODEL",
    "gpt-5.4-nano",
).strip()
if not OPENAI_WEB_SEARCH_MODEL:
    raise ValueError("OPENAI_WEB_SEARCH_MODEL must not be empty")
OPENAI_WEB_SEARCH_REASONING_EFFORT = os.getenv(
    "OPENAI_WEB_SEARCH_REASONING_EFFORT",
    "low",
).strip().casefold()
if OPENAI_WEB_SEARCH_REASONING_EFFORT not in {
    "none", "low", "medium", "high", "xhigh",
}:
    raise ValueError(
        "OPENAI_WEB_SEARCH_REASONING_EFFORT must be one of: "
        "none, low, medium, high, xhigh"
    )
OPENAI_WEB_SEARCH_CONTEXT_SIZE = os.getenv(
    "OPENAI_WEB_SEARCH_CONTEXT_SIZE",
    "low",
).strip().casefold()
if OPENAI_WEB_SEARCH_CONTEXT_SIZE not in {"low", "medium", "high"}:
    raise ValueError(
        "OPENAI_WEB_SEARCH_CONTEXT_SIZE must be one of: low, medium, high"
    )
OPENAI_WEB_SEARCH_COUNTRY = os.getenv(
    "OPENAI_WEB_SEARCH_COUNTRY",
    "UA",
).strip().upper()
if (
    len(OPENAI_WEB_SEARCH_COUNTRY) != 2
    or not OPENAI_WEB_SEARCH_COUNTRY.isascii()
    or not OPENAI_WEB_SEARCH_COUNTRY.isalpha()
):
    raise ValueError("OPENAI_WEB_SEARCH_COUNTRY must be two ASCII letters")
OPENAI_WEB_SEARCH_MAX_RESULTS = _environment_integer(
    "OPENAI_WEB_SEARCH_MAX_RESULTS", "5", 1, 10
)
OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS = _environment_integer(
    "OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS", "1024", 256, 4096
)
OPENAI_WEB_SEARCH_TIMEOUT_SECONDS = _environment_float(
    "OPENAI_WEB_SEARCH_TIMEOUT_SECONDS", "20", 0.0, 30.0
)
OPENAI_WEB_SEARCH_EXTERNAL_ACCESS = (
    _environment_boolean("OPENAI_WEB_SEARCH_EXTERNAL_ACCESS", "true") == "true"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
OPENAI_SCORING_ENABLED = _environment_boolean("OPENAI_SCORING_ENABLED", "false")
OPENAI_SCORING_REASONING_EFFORT = os.getenv(
    "OPENAI_SCORING_REASONING_EFFORT",
    "minimal",
).strip().casefold()
if OPENAI_SCORING_REASONING_EFFORT not in {"minimal", "low", "medium", "high"}:
    raise ValueError(
        "OPENAI_SCORING_REASONING_EFFORT must be one of: minimal, low, medium, high"
    )
OPENAI_SCORING_MAX_OUTPUT_TOKENS = _environment_integer(
    "OPENAI_SCORING_MAX_OUTPUT_TOKENS",
    "512",
    64,
    1024,
)
OPENAI_SCORING_TIMEOUT_SECONDS = _environment_float(
    "OPENAI_SCORING_TIMEOUT_SECONDS",
    "20",
    0.0,
    60.0,
)

# --- Прогресс ---
PROGRESS_INTERVAL = 180         # отправлять прогресс каждые 3 минуты (180 сек)


# --- Доступ пользователей ---
def _environment_user_ids(name: str) -> set[int]:
    """Список Telegram user_id через запятую. Пусто = доступ разрешён всем."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    ids: set[int] = set()
    for chunk in re.split(r"[,\s;]+", raw):
        if not chunk:
            continue
        if re.fullmatch(r"[0-9]+", chunk) is None:
            raise ValueError(f"{name} must contain only numeric Telegram user IDs")
        ids.add(int(chunk))
    return ids


# Если ALLOWED_USER_IDS пуст — ботом может пользоваться любой пользователь.
ALLOWED_USER_IDS = _environment_user_ids("ALLOWED_USER_IDS")

# Сколько поисков может выполняться одновременно (разные пользователи).
MAX_CONCURRENT_SEARCHES = _environment_integer("MAX_CONCURRENT_SEARCHES", "3", 1, 20)

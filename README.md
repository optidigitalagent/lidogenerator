# Lead Hunter Bot

Telegram-бот, который ищет бизнесы без сайтов в Google Maps (Украина):
собирает контакты, проверяет сайты и Instagram, оценивает лиды через Claude AI
и отдаёт готовую CSV-таблицу.

## Установка — 5 шагов

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Установить браузер для парсинга
playwright install chromium

# 3. Создать файл настроек
copy .env.example .env        # Linux/Mac: cp .env.example .env

# 4. Вписать в .env токен бота (получить у @BotFather в Telegram)
#    TELEGRAM_TOKEN=12345:AAA...
#    ANTHROPIC_API_KEY можно не указывать — тогда AI-скоринг отключён (score=50)

# 5. Запустить
python bot.py
```

## Как пользоваться

В Telegram напиши боту:

| Команда   | Что делает                                          |
|-----------|-----------------------------------------------------|
| `/start`  | Приветствие и инструкция                            |
| `/search` | Новый поиск: ниша → город → количество → запуск     |
| `/status` | Статус текущего поиска                              |
| `/export` | Скачать CSV последнего поиска                       |
| `/stop`   | Остановить текущий поиск (собранное сохраняется)    |
| `/sync`   | Показать очередь Opti и повторить ожидающие доставки |

Поиск 50–200 бизнесов занимает 30–90 минут. Прогресс приходит в чат каждые 3 минуты.

## Структура проекта

```
lidogenerator/
├── bot.py                  # Telegram-бот — точка входа
├── orchestrator.py         # цепочка агентов: сбор → проверки → AI → CSV
├── config.py               # настройки и чтение .env
├── models.py               # датакласс Business
├── db.py                   # SQLite: задачи и бизнесы
├── requirements.txt
├── .env.example            # шаблон настроек
├── agents/
│   ├── collector.py        # парсинг Google Maps (Playwright)
│   ├── site_checker.py     # проверка сайтов (httpx)
│   ├── social_checker.py   # проверка Instagram (Playwright)
│   ├── ai_scorer.py        # AI-оценка лидов (Claude API, claude-haiku-4-5)
│   └── reporter.py         # экспорт CSV (UTF-8 BOM, сортировка по score)
├── tests/                  # пошаговые тесты этапов 2–8
└── exports/                # готовые CSV-файлы
```

## Стоимость

Всё бесплатно, кроме AI-скоринга: ~$0.30 на 100 бизнесов (Claude Haiku).
Без ключа `ANTHROPIC_API_KEY` система работает в режиме фильтрации без AI.

## Opti Bridge v0

Lead Generator remains the external discovery source; Opti is the CRM and the
source of truth after import. After a completed search has persisted its final
qualified rows and completed the existing CSV/XLSX export, the bridge builds
`opti.lead-import.v1` directly from SQLite, enqueues immutable JSON in
`opti_sync_outbox`, and makes one best-effort delivery. It never parses exports,
and an unavailable Opti service does not fail the search or export.

Delivery is disabled by default. Set `OPTI_BRIDGE_ENABLED=true`, an HTTP(S)
`OPTI_BASE_URL`, and the server-only `OPTI_IMPORT_TOKEN`. Requests go to
`POST /integrations/lead-generator/import-batches` with a bearer token and the
persisted idempotency key
`lidogenerator:<externalBatchId>:<payloadHash-prefix>`. Redirects are rejected,
timeouts are bounded, and there is no HTTP-layer automatic retry.

The SQLite worker recovers interrupted `SENDING` rows at startup. Transport and
temporary server failures move rows to `RETRY` with capped exponential backoff.
Authentication, configuration, redirect, request, and payload-conflict failures
move rows to `FAILED`; `/sync` resets failed rows for an explicit retry and shows
only counts, never tokens or payload bodies. The maximum attempts and retry base
are controlled by `OPTI_SYNC_MAX_ATTEMPTS` and
`OPTI_SYNC_RETRY_BASE_SECONDS`.

The stable lead identity priority is: explicit upstream candidate ID, Google
Place ID, normalized Maps URL, normalized phone, Instagram handle, website
domain, then a versioned SHA-256 of normalized name/city/address. Local SQLite
autoincrement IDs are not used. Completed searches enqueue once. Stopped searches
are not imported in v0 because the current stop path persists a partial set
before final enrichment/export stabilization.

The canonical example is
`contracts/opti-lead-import-v1.example.json`. For a local, provider-free smoke,
configure a disposable `DB_PATH`, run `python scripts/enqueue_opti_smoke.py`,
then start the bot worker or use `/sync`. The helper only enqueues a fixed
three-lead batch and performs no network call by itself.

## Opti Control API v0

Telegram remains fully supported. When explicitly enabled, the bot process also
serves a small authenticated HTTP API used by the Opti web workspace. Telegram
and Opti searches share one in-process `SearchRuntime`, stop-event registry, and
`MAX_CONCURRENT_SEARCHES` semaphore. Interrupted nonterminal searches are marked
`error` with `PROCESS_RESTARTED` at startup; they are not falsely resumed.

Configuration:

```text
LEAD_GENERATOR_CONTROL_ENABLED=false
LEAD_GENERATOR_CONTROL_TOKEN=
HOST=0.0.0.0
PORT=8080
```

When enabled, the token must contain at least 32 characters or startup fails
closed. `/health` is public and minimal. All routes under `/internal/opti/v1`
require a bearer token compared in constant time. See
`contracts/opti-control-api-v0.md` for the route and DTO contract.

The browser never receives this control token: it talks only to Opti's
authenticated same-origin API, and Opti calls this API server-to-server. Search
tasks, structured progress, terminal state, idempotency, and the outbox remain in
Lead Generator SQLite only. Opti does not mirror them. Completed leads still
reach Opti exclusively through the existing Bridge and immutable outbox.

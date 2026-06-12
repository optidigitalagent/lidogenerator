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

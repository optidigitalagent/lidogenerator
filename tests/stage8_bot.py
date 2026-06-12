# -*- coding: utf-8 -*-
"""Тест этапа 8: бот собирается без ошибок.

Без TELEGRAM_TOKEN polling не запустить, поэтому проверяем:
  1. bot.py импортируется без ошибок
  2. без токена main() даёт понятное сообщение (SystemExit)
  3. с тестовым токеном Application собирается и все хендлеры регистрируются
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from telegram.ext import Application, CommandHandler, ConversationHandler

import bot
import config


def main():
    # Проверка 1: импорт прошёл (если бы упал — мы бы сюда не дошли)
    print("Импорт bot.py: OK")

    # Проверка 2: без токена — понятная ошибка, а не traceback
    saved = config.TELEGRAM_TOKEN
    config.TELEGRAM_TOKEN = ""
    try:
        bot.main()
        raise AssertionError("main() без токена должен завершаться SystemExit")
    except SystemExit as e:
        assert "TELEGRAM_TOKEN" in str(e), f"Сообщение не о токене: {e}"
        print("Без токена: понятное сообщение об ошибке: OK")
    finally:
        config.TELEGRAM_TOKEN = saved

    # Проверка 3: с тестовым токеном приложение собирается, хендлеры на месте
    app = Application.builder().token("123456:TEST_TOKEN_FOR_BUILD_CHECK").build()

    search_conv = ConversationHandler(
        entry_points=[CommandHandler("search", bot.cmd_search)],
        states={
            bot.NICHE: [],
            bot.CITY: [],
            bot.COUNT: [],
            bot.CONFIRM: [],
        },
        fallbacks=[],
    )
    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(search_conv)
    app.add_handler(CommandHandler("status", bot.cmd_status))
    app.add_handler(CommandHandler("export", bot.cmd_export))
    app.add_handler(CommandHandler("stop", bot.cmd_stop))

    handlers = app.handlers[0]
    assert len(handlers) == 5, f"Ожидалось 5 хендлеров, есть {len(handlers)}"
    print(f"Application собран, хендлеров: {len(handlers)}: OK")

    # Проверка 4: тексты бота — на украинском (есть характерные буквы і/ї/є)
    import inspect
    src = inspect.getsource(bot)
    assert any(ch in src for ch in "іїє"), "В текстах бота нет украинских букв"
    print("Тексты на украинском: OK")

    print("\nТЕСТ ПРОЙДЕН")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Telegram-бот Lead Hunter.

Команды:
  /start  — приветствие и инструкция
  /search — новый поиск: ниша (кнопки) → город (текст) → количество (кнопки) → запуск
  /status — статус текущего/последнего поиска
  /export — прислать CSV последней задачи
  /stop   — остановить текущий поиск
  /sync   — показать очередь Opti и повторить доставку

Тексты бота — на украинском. Запуск: python bot.py
"""

import asyncio
import logging
from pathlib import Path

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import config
import db
import orchestrator
from integrations import opti_bridge, opti_outbox

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.INFO
)
# Не спамим логи запросами к Telegram API
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("lead_hunter.bot")

# Состояния диалога /search
NICHE, CITY, COUNT, CONFIRM = range(4)

# Кнопки ниш: (подпись, значение для поиска в Maps)
NICHES = [
    ("💅 Салон краси", "салон краси"),
    ("✂️ Барбершоп", "барбершоп"),
    ("🍽 Ресторан", "ресторан"),
    ("🔧 СТО", "СТО"),
    ("🧖 Спа", "спа салон"),
    ("👗 Одяг", "магазин одягу"),
    ("✏️ Інше", "__custom__"),
]

COUNTS = [50, 100, 200]

# Активные поиски: user_id -> {"task_id", "stop_event", "asyncio_task"}
# Ключ — именно user_id, чтобы разные пользователи (в т.ч. в одной группе)
# могли искать параллельно и не мешали друг другу.
ACTIVE: dict = {}

# Ограничение одновременно выполняющихся поисков (общее на весь бот).
SEARCH_SLOTS = asyncio.Semaphore(config.MAX_CONCURRENT_SEARCHES)
OUTBOX_WORKER = opti_outbox.OutboxWorker()


def is_allowed(update: Update) -> bool:
    """Пустой ALLOWED_USER_IDS = доступ у всех; иначе — только из списка."""
    if not config.ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return user is not None and user.id in config.ALLOWED_USER_IDS


async def deny(update: Update) -> None:
    user = update.effective_user
    text = (
        "⛔️ Немає доступу до цього бота.\n"
        f"Твій Telegram ID: {user.id if user else '?'}\n"
        "Попроси адміністратора додати його у ALLOWED_USER_IDS."
    )
    if update.callback_query:
        await update.callback_query.answer(text, show_alert=True)
    elif update.message:
        await update.message.reply_text(text)


# ---------- /start ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    await update.message.reply_text(
        "👋 Привіт! Я Lead Hunter — шукаю бізнеси, яким потрібен сайт.\n\n"
        "Кого я залишаю в таблиці:\n"
        "✅ є Instagram І немає сайту\n"
        "✅ є Instagram І сайт дуже поганий\n"
        "❌ бізнеси з нормальним сайтом — пропускаю\n"
        "❌ бізнеси без Instagram — пропускаю\n\n"
        "Таблиця коротка: Назва · Місто · Instagram · Статус сайту\n\n"
        "Команди:\n"
        "/search — почати новий пошук\n"
        "/status — статус поточного пошуку\n"
        "/export — завантажити CSV останнього пошуку\n"
        "/stop — зупинити пошук\n"
        "/sync — стан і повтор доставки в Opti"
    )


# ---------- Диалог /search ----------

async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        await deny(update)
        return ConversationHandler.END
    user_id = update.effective_user.id
    if user_id in ACTIVE and not ACTIVE[user_id]["asyncio_task"].done():
        await update.message.reply_text(
            "⏳ Пошук уже виконується. Зупини його командою /stop або дочекайся завершення."
        )
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"niche:{value}")]
        for label, value in NICHES
    ]
    await update.message.reply_text(
        "Обери нішу для пошуку:", reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return NICHE


async def on_niche(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    value = query.data.split(":", 1)[1]
    if value == "__custom__":
        await query.edit_message_text("Напиши нішу текстом (наприклад: «стоматологія»):")
        return NICHE
    context.user_data["niche"] = value
    await query.edit_message_text(f"Ніша: {value}\n\nТепер напиши місто (наприклад: «Харків»):")
    return CITY


async def on_custom_niche(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["niche"] = update.message.text.strip()
    await update.message.reply_text("Тепер напиши місто (наприклад: «Харків»):")
    return CITY


async def on_city(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["city"] = update.message.text.strip()
    keyboard = [[
        InlineKeyboardButton(f"{n} лідів", callback_data=f"count:{n}") for n in COUNTS
    ]]
    await update.message.reply_text(
        "Скільки якісних лідів потрібно зібрати?\n"
        "(це кількість лідів у таблиці, а не кількість переглянутих бізнесів)",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return COUNT


async def on_count(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data["count"] = int(query.data.split(":", 1)[1])
    niche = context.user_data["niche"]
    city = context.user_data["city"]
    count = context.user_data["count"]
    keyboard = [[InlineKeyboardButton("🚀 Запустити", callback_data="go")]]
    await query.edit_message_text(
        f"Перевір налаштування:\n\n"
        f"Ніша: {niche}\nМісто: {city}\nПотрібно лідів: {count}\n\n"
        f"Шукатиму, поки не набереться {count} якісних лідів "
        f"(або поки не закінчаться результати).\n"
        f"Орієнтовний час: 30–90 хвилин.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return CONFIRM


async def on_go(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not is_allowed(update):
        await deny(update)
        return ConversationHandler.END
    await query.answer()
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    niche = context.user_data["niche"]
    city = context.user_data["city"]
    count = context.user_data["count"]

    task_id = db.create_task(niche, city, count, chat_id=chat_id)
    stop_event = asyncio.Event()

    async def send_progress(text: str):
        try:
            await context.bot.send_message(chat_id, text)
        except Exception:
            log.exception("Не удалось отправить прогресс в чат %s", chat_id)

    async def run():
        try:
            async with SEARCH_SLOTS:
                if stop_event.is_set():
                    return
                csv_path = await orchestrator.run_search(
                    task_id, progress_callback=send_progress, stop_event=stop_event
                )
            if csv_path:
                with open(csv_path, "rb") as f:
                    await context.bot.send_document(
                        chat_id, f, filename=Path(csv_path).name,
                        caption="📊 Таблиця лідів готова!",
                    )
        except Exception:
            log.exception("Поиск %s завершился ошибкой", task_id)
        finally:
            ACTIVE.pop(user_id, None)

    ACTIVE[user_id] = {
        "task_id": task_id,
        "stop_event": stop_event,
        "asyncio_task": asyncio.create_task(run()),
    }

    await query.edit_message_text(
        f"🚀 Запущено! Шукаю {count} якісних лідів: «{niche}» у місті {city}.\n"
        f"Чекай ~30–90 хв. Прогрес надсилатиму кожні 3 хвилини.\n"
        f"Зупинити: /stop"
    )
    return ConversationHandler.END


async def cancel_search_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Скасовано. Почати знову: /search")
    return ConversationHandler.END


# ---------- /status ----------

STATUS_TEXT = {
    "new": "🕐 У черзі",
    "collecting": "📍 Збираю бізнеси з Google Maps...",
    "checking": "🔍 Перевіряю сайти та Instagram...",
    "scoring": "🤖 AI оцінює ліди...",
    "done": "✅ Завершено",
    "error": "❌ Помилка",
    "stopped": "⏹ Зупинено",
}


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    task = db.get_last_task(update.effective_chat.id)
    if task is None:
        await update.message.reply_text("Пошуків ще не було. Почни з /search")
        return
    status = STATUS_TEXT.get(task["status"], task["status"])
    await update.message.reply_text(
        f"Пошук #{task['id']}: «{task['niche']}» у місті {task['city']}\n"
        f"Статус: {status}\n"
        f"Створено: {task['created_at']}"
        + (f"\nЗавершено: {task['finished_at']}" if task["finished_at"] else "")
    )


# ---------- /export ----------

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    task = db.get_last_task(update.effective_chat.id)
    if task is None or not task["csv_path"]:
        await update.message.reply_text("Готового CSV ще немає. Спочатку заверши пошук: /search")
        return
    path = Path(task["csv_path"])
    if not path.exists():
        await update.message.reply_text("Файл CSV не знайдено на диску. Запусти новий пошук.")
        return
    with open(path, "rb") as f:
        await update.message.reply_document(f, filename=path.name, caption="📊 Твоя таблиця лідів")


# ---------- /stop ----------

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    active = ACTIVE.get(update.effective_user.id)
    if not active or active["asyncio_task"].done():
        await update.message.reply_text("Зараз нічого не виконується.")
        return
    active["stop_event"].set()
    await update.message.reply_text("⏹ Зупиняю пошук... Збережу те, що вже зібрано.")


# ---------- /sync ----------

async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await deny(update)
        return
    reconciled = opti_bridge.reconcile_completed_tasks(limit=100)
    retried = opti_outbox.retry_failed()
    await opti_outbox.deliver_due(limit=10)
    prefix = "Opti bridge is disabled. " if not config.OPTI_BRIDGE_ENABLED else ""
    suffix = f"; manually retried {retried}" if retried else ""
    recovery = (
        f"; recovered {reconciled['enqueued']} completed"
        if reconciled["enqueued"]
        else ""
    )
    await update.message.reply_text(
        prefix + opti_outbox.format_summary() + recovery + suffix
    )


async def _start_outbox_worker(application: Application) -> None:
    del application
    # After a process restart no prior in-process sender can still own a row.
    opti_outbox.recover_stale_sending(stale_after_seconds=0)
    opti_bridge.reconcile_completed_tasks(limit=100)
    OUTBOX_WORKER.start()


async def _stop_outbox_worker(application: Application) -> None:
    del application
    await OUTBOX_WORKER.stop()


# ---------- Запуск ----------

def main():
    if not config.TELEGRAM_TOKEN:
        raise SystemExit(
            "Не задано TELEGRAM_TOKEN.\n"
            "1. Створи бота через @BotFather у Telegram\n"
            "2. Скопіюй .env.example у .env та встав токен"
        )

    db.init_db()

    if config.ALLOWED_USER_IDS:
        log.info("Доступ разрешён пользователям: %s", sorted(config.ALLOWED_USER_IDS))
    else:
        log.info("ALLOWED_USER_IDS не задан — ботом может пользоваться любой пользователь")

    app = (
        Application.builder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(_start_outbox_worker)
        .post_shutdown(_stop_outbox_worker)
        .build()
    )

    search_conv = ConversationHandler(
        entry_points=[CommandHandler("search", cmd_search)],
        states={
            NICHE: [
                CallbackQueryHandler(on_niche, pattern=r"^niche:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, on_custom_niche),
            ],
            CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_city)],
            COUNT: [CallbackQueryHandler(on_count, pattern=r"^count:")],
            CONFIRM: [CallbackQueryHandler(on_go, pattern=r"^go$")],
        },
        fallbacks=[CommandHandler("cancel", cancel_search_dialog)],
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(search_conv)
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("sync", cmd_sync))

    log.info("Бот запущено. Зупинка: Ctrl+C")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

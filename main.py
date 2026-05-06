# pylint: disable=unused-argument
"""
Simple and easy to deploy telegram bot for communication with LLM.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import os

import httpx
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    PicklePersistence,
    filters,
)
from dotenv import dotenv_values


class InputMode(Enum):
    TEXT = 1
    DOCUMENTS = 2
    IMAGES = 3


class OutputVerbosity(Enum):
    SHORT = 1
    DEFAULT = 2
    VERBOSE = 3


@dataclass
class LLMChatSettings:
    input_mode: InputMode | None
    output_verbosity: OutputVerbosity | None


@dataclass
class LLMChat:
    settings: LLMChatSettings
    data: dict


class BotState(Enum):
    INPUT_MODE_SELECTION = 1
    OUTPUT_VERBOSITY_SELECTION = 2
    CHATTING_WITH_LLM = 3


CHAT_COMPLETIONS_API_PATH = "/chat/completions"

ENV_FILE_PATH = ".env"
CONFIG: dict

TELEGRAM_BOT_TOKEN: str
TELEGRAM_BOT_PERSISTENCE_FILE: str
LLM_API_URL: str

llm_client: httpx.AsyncClient
logger: logging.Logger


def check_env_file(path: str) -> None:
    """Check env file existence."""
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} does not exist or is not a file")


def check_config(config: dict) -> None:
    """Check required variables in config."""
    if "TELEGRAM_BOT_TOKEN" not in config:
        raise RuntimeError(f"TELEGRAM_BOT_TOKEN is not provided in {ENV_FILE_PATH}")

    if "TELEGRAM_BOT_PERSISTENCE_FILE" not in config:
        raise RuntimeError(
            f"TELEGRAM_BOT_PERSISTENCE_FILE is not provided in {ENV_FILE_PATH}"
        )

    if "LLM_API_URL" not in config:
        raise RuntimeError(f"LLM_API_URL is not provided in {ENV_FILE_PATH}")


def get_input_mode(s: str) -> InputMode:
    match s:
        case "text":
            return InputMode.TEXT
        case "documents":
            return InputMode.DOCUMENTS
        case "images":
            return InputMode.IMAGES
        case _:
            raise RuntimeError("`s` is unknown")


def get_output_verbosity(s: str) -> OutputVerbosity:
    match s:
        case "short":
            return OutputVerbosity.SHORT
        case "default":
            return OutputVerbosity.DEFAULT
        case "verbose":
            return OutputVerbosity.VERBOSE
        case _:
            raise RuntimeError("`s` is unknown")


async def input_mode_selection_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    keyboard = [
        [InlineKeyboardButton("Работа с текстом", callback_data="text")],
        [InlineKeyboardButton("Работа с документами", callback_data="documents")],
        [InlineKeyboardButton("Работа с изображениями", callback_data="images")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите режим работы:", reply_markup=reply_markup)


async def output_verbosity_selection_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user = update.effective_user
    keyboard = [
        [InlineKeyboardButton("Краткий", callback_data="short")],
        [InlineKeyboardButton("Стандартный", callback_data="default")],
        [InlineKeyboardButton("Подробный", callback_data="verbose")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await user.send_message(
        "Выберите степень подробности ответа:", reply_markup=reply_markup
    )


async def chatting_with_llm_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user = update.effective_user
    await user.send_message(
        "Чат с LLM создан. Далее пишите ваши промпты. "
        "Для завершения чата используйте команду /stop."
    )


async def input_mode_selection_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> BotState:
    query = update.callback_query

    context.user_data["llm_chat"].settings.input_mode = get_input_mode(query.data)

    await query.answer()
    await query.edit_message_text(f"Выбранный режим работы: {query.data}.")

    await output_verbosity_selection_entry(update, context)
    return BotState.OUTPUT_VERBOSITY_SELECTION


async def output_verbosity_selection_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> BotState:
    query = update.callback_query

    context.user_data["llm_chat"].settings.output_verbosity = get_output_verbosity(
        query.data
    )

    await query.answer()
    await query.edit_message_text(f"Выбранный режим работы: {query.data}.")

    await chatting_with_llm_entry(update, context)
    return BotState.CHATTING_WITH_LLM


async def get_next_llm_chat_completion(data: dict) -> str:
    response = await llm_client.post(
        CHAT_COMPLETIONS_API_PATH,
        json=data,
    )
    response.raise_for_status()
    response_data = response.json()

    completions = []
    for choice in response_data["choices"]:
        completions.append(choice)

    if len(completions) > 1:
        raise RuntimeError("LLM returned more than one choice")

    if completions[0]["finish_reason"] != "stop":
        raise RuntimeError("LLM finish reason is not `stop`")

    return completions[0]["message"]["content"]


def get_system_prompt(settings: LLMChatSettings) -> str:
    # TODO: create system prompt selection logic

    # Telegram message can have 4096 characters at max so specify this in
    # system prompt
    return (
        "You are a helpful assistant.\n"
        "Your output must always be less or equal to 4096 characters.\n"
    )


async def chatting_with_llm_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    llm_chat = context.user_data["llm_chat"]

    if not llm_chat.data:
        llm_chat.data = {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt(llm_chat.settings),
                }
            ]
        }

    prompt = update.message.text
    llm_messages = llm_chat.data["messages"]
    llm_messages.append({"role": "user", "content": prompt})

    bot_message = await update.message.reply_text("Думаю...")
    try:
        answer = await get_next_llm_chat_completion(llm_chat.data)
        llm_messages.append({"role": "assistant", "content": answer})
        await bot_message.edit_text(answer)
    except Exception as ex:
        await bot_message.edit_text("Ошибка при взаимодействии с LLM.")
        raise ex


async def llm_configuration_unsupported_command_message(
    command: str, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        f"Команда /{command} недоступна во время создания чата чата.\n"
        "Чтобы завершить создание чата используйте команду /cancel."
    )


async def llm_configuration_cancel_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Send a message when the command /cancel is issued."""
    await update.message.reply_text("Создание чата отменено.")
    return ConversationHandler.END


async def llm_configuration_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /start is issued while configuring LLM."""
    await llm_configuration_unsupported_command_message("start", update, context)


async def llm_configuration_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /new is issued while configuring LLM."""
    await llm_configuration_unsupported_command_message("new", update, context)


def generate_help(data: dict[str, str]) -> str:
    rows = []
    for command, description in data.items():
        rows.append(f"/{command} - {description}.")
    return "\n".join(rows)


async def llm_configuration_help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued while configuring LLM."""
    await update.message.reply_text(
        generate_help(
            {"help": "Вывести это сообщение", "cancel": "Отменить создание чата"}
        )
    )


async def chatting_with_llm_unsupported_command_message(
    command: str, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        f"Команда /{command} недоступна во время чата с LLM.\n"
        "Чтобы завершить чат с LLM используйте команду /stop."
    )


async def chatting_with_llm_stop_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Send a message when the command /stop is issued."""
    await update.message.reply_text("Чат завершён.")
    context.user_data["llm_chat"].settings = None
    context.user_data["llm_chat"].data = None
    return ConversationHandler.END


async def chatting_with_llm_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /start is issued while chatting with LLM."""
    await chatting_with_llm_unsupported_command_message("start", update, context)


async def chatting_with_llm_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /new is issued while configuring LLM."""
    await chatting_with_llm_unsupported_command_message("new", update, context)


async def chatting_with_llm_help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued while chatting with LLM."""
    await update.message.reply_text(
        generate_help({"help": "Вывести это сообщение", "stop": "Завершить чат с LLM"})
    )


async def entry_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> BotState:
    """Send a message when the command /new is issued."""
    context.user_data["llm_chat"] = LLMChat(LLMChatSettings(None, None), {})
    await input_mode_selection_entry(update, context)
    return BotState.INPUT_MODE_SELECTION


async def global_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        f"Привет, {update.effective_user.first_name}!\n"
        "Используйте команду /new чтобы начать новый чат."
    )


async def global_help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued when not in a
    conversation."""
    await update.message.reply_text(
        generate_help(
            {
                "start": "Вывести первоначальное сообщение",
                "help": "Вывести это сообщение",
                "new": "Начать новый чат",
            }
        )
    )


async def global_unknown_command_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when an unknown command is issued."""
    await update.message.reply_text(
        "Введена неизвестная команда.\n"
        "Используйте команду /help, чтобы посмотреть список доступных команд."
    )


def main() -> None:
    """Start the bot."""
    persistence = PicklePersistence(filepath=TELEGRAM_BOT_PERSISTENCE_FILE)
    application = (
        Application.builder().token(TELEGRAM_BOT_TOKEN).persistence(persistence).build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("new", entry_new_handler)],
        states={
            BotState.INPUT_MODE_SELECTION: [
                CallbackQueryHandler(input_mode_selection_handler),
                CommandHandler("cancel", llm_configuration_cancel_handler),
                CommandHandler("start", llm_configuration_start_handler),
                CommandHandler("new", llm_configuration_new_handler),
                CommandHandler("help", llm_configuration_help_handler),
            ],
            BotState.OUTPUT_VERBOSITY_SELECTION: [
                CallbackQueryHandler(output_verbosity_selection_handler),
                CommandHandler("cancel", llm_configuration_cancel_handler),
                CommandHandler("start", llm_configuration_start_handler),
                CommandHandler("new", llm_configuration_new_handler),
                CommandHandler("help", llm_configuration_help_handler),
            ],
            BotState.CHATTING_WITH_LLM: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, chatting_with_llm_handler
                ),
                CommandHandler("stop", chatting_with_llm_stop_handler),
                CommandHandler("start", chatting_with_llm_start_handler),
                CommandHandler("new", chatting_with_llm_new_handler),
                CommandHandler("help", chatting_with_llm_help_handler),
            ],
        },
        fallbacks=[],
        name="main_conversation_handler",
        persistent=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("start", global_start_handler))
    application.add_handler(CommandHandler("help", global_help_handler))
    application.add_handler(
        MessageHandler(filters.COMMAND, global_unknown_command_handler)
    )

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    check_env_file(ENV_FILE_PATH)

    CONFIG = dotenv_values(ENV_FILE_PATH)
    check_config(CONFIG)

    TELEGRAM_BOT_TOKEN = CONFIG["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_BOT_PERSISTENCE_FILE = CONFIG["TELEGRAM_BOT_PERSISTENCE_FILE"]
    LLM_API_URL = CONFIG["LLM_API_URL"]

    llm_client = httpx.AsyncClient(
        base_url=LLM_API_URL,
        timeout=httpx.Timeout(300.0),
    )

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    main()

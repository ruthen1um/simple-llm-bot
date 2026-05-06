# pylint: disable=unused-argument
"""
Simple and easy to deploy telegram bot for communication with LLM.
"""
import logging
import os
from enum import Enum
from dataclasses import dataclass

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

from quickllmbot import llm
from quickllmbot import strings


class ConversationState(Enum):
    """Represents the conversation state."""

    LLM_MODE_SELECTION = 1
    LLM_VERBOSITY_SELECTION = 2
    LLM_CHATTING = 3


CHAT_COMPLETIONS_API_PATH = "/chat/completions"

LLM_CHAT_USER_DATA_FIELD = "llm_chat"

@dataclass(frozen=True)
class CallbackData:
    LLM_MODE_TEXT = "text"
    LLM_MODE_DOCUMENTS = "documents"
    LLM_MODE_IMAGES = "images"

    LLM_VERBOSITY_SHORT = "short"
    LLM_VERBOSITY_DEFAULT = "default"
    LLM_VERBOSITY_VERBOSE = "verbose"

ENV_FILE_PATH = ".env"
CONFIG: dict

TELEGRAM_BOT_TOKEN: str
TELEGRAM_BOT_PERSISTENCE_FILE: str
LLM_API_URL: str

llm_client: httpx.AsyncClient
logger: logging.Logger

def check_env_file(path: str) -> None:
    """Checks env file existence."""
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} does not exist or is not a file")


def check_config(config: dict) -> None:
    """Checks required variables existence in config."""
    if "TELEGRAM_BOT_TOKEN" not in config:
        raise RuntimeError(f"TELEGRAM_BOT_TOKEN is not provided in {ENV_FILE_PATH}")

    if "TELEGRAM_BOT_PERSISTENCE_FILE" not in config:
        raise RuntimeError(
            f"TELEGRAM_BOT_PERSISTENCE_FILE is not provided in {ENV_FILE_PATH}"
        )

    if "LLM_API_URL" not in config:
        raise RuntimeError(f"LLM_API_URL is not provided in {ENV_FILE_PATH}")


def get_llm_mode(s: str) -> llm.LLMMode:
    """Returns LLMMode object based on input string"""
    match s:
        case CallbackData.LLM_MODE_TEXT:
            return llm.LLMMode.TEXT
        case CallbackData.LLM_MODE_DOCUMENTS:
            return llm.LLMMode.DOCUMENTS
        case CallbackData.LLM_MODE_IMAGES:
            return llm.LLMMode.IMAGES
        case _:
            raise ValueError("`s` does not correspond to a valid LLMMode object")


def get_llm_verbosity(s: str) -> llm.LLMVerbosity:
    """Returns LLMVerbosity object based on input string"""
    match s:
        case CallbackData.LLM_VERBOSITY_SHORT:
            return llm.LLMVerbosity.SHORT
        case CallbackData.LLM_VERBOSITY_DEFAULT:
            return llm.LLMVerbosity.DEFAULT
        case CallbackData.LLM_VERBOSITY_VERBOSE:
            return llm.LLMVerbosity.VERBOSE
        case _:
            raise ValueError("`s` does not correspond to a valid LLMVerbosity object")


async def llm_mode_selection_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Sends message upon entering LLM_MODE_SELECTION conversation state."""
    keyboard = [
        [
            InlineKeyboardButton(
                strings.LLMModeSelection.TEXT_MODE_BUTTON,
                callback_data=CallbackData.LLM_MODE_TEXT,
            )
        ],
        [
            InlineKeyboardButton(
                strings.LLMModeSelection.DOCUMENTS_MODE_BUTTON,
                callback_data=CallbackData.LLM_MODE_DOCUMENTS,
            )
        ],
        [
            InlineKeyboardButton(
                strings.LLMModeSelection.IMAGES_MODE_BUTTON,
                callback_data=CallbackData.LLM_MODE_IMAGES,
            )
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        strings.LLMModeSelection.REQUEST, reply_markup=reply_markup
    )


async def llm_verbosity_selection_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Sends message upon entering LLM_VERBOSITY_SELECTION conversation state."""
    user = update.effective_user
    keyboard = [
        [
            InlineKeyboardButton(
                strings.LLMVerbositySelection.SHORT_VERBOSITY_BUTTON,
                callback_data=CallbackData.LLM_VERBOSITY_SHORT,
            )
        ],
        [
            InlineKeyboardButton(
                strings.LLMVerbositySelection.DEFAULT_VERBOSITY_BUTTON,
                callback_data=CallbackData.LLM_VERBOSITY_DEFAULT,
            )
        ],
        [
            InlineKeyboardButton(
                strings.LLMVerbositySelection.VERBOSE_VERBOSITY_BUTTON,
                callback_data=CallbackData.LLM_VERBOSITY_VERBOSE,
            )
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await user.send_message(
        strings.LLMVerbositySelection.REQUEST, reply_markup=reply_markup
    )


async def llm_chatting_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user = update.effective_user
    await user.send_message(strings.Global.CHAT_CREATION_SUCCESS)


async def llm_mode_selection_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> ConversationState:
    query = update.callback_query
    data = query.data

    context.user_data[LLM_CHAT_USER_DATA_FIELD].settings.mode = get_llm_mode(data)

    await query.answer()
    await query.edit_message_text(
        strings.LLMModeSelection.SELECTION_FORMAT.format(mode=data)
    )

    await llm_verbosity_selection_entry(update, context)
    return ConversationState.LLM_VERBOSITY_SELECTION


async def llm_verbosity_selection_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> ConversationState:
    query = update.callback_query
    data = query.data

    context.user_data[LLM_CHAT_USER_DATA_FIELD].settings.verbosity = get_llm_verbosity(
        data
    )

    await query.answer()
    await query.edit_message_text(
        strings.LLMVerbositySelection.SELECTION_FORMAT.format(verbosity=data)
    )

    await llm_chatting_entry(update, context)
    return ConversationState.LLM_CHATTING


async def get_next_llm_chat_completion(data: dict) -> str:
    response = await llm_client.post(
        CHAT_COMPLETIONS_API_PATH,
        json=data,
    )
    response.raise_for_status()
    response_data = response.json()

    completions = []
    # TODO: make more efficient by just counting
    for choice in response_data["choices"]:
        completions.append(choice)

    if len(completions) > 1:
        raise RuntimeError("LLM returned more than one choice")

    if completions[0]["finish_reason"] != "stop":
        raise RuntimeError("LLM finish reason is not `stop`")

    return completions[0]["message"]["content"]


def get_system_prompt(settings: llm.LLMSettings) -> str:
    """Returns system prompt corresponding to provided settings."""
    # TODO: implement system prompt selection logic
    # TODO: figure out what to do if LLM output does not fit in 4096 character limit

    # Telegram message can have 4096 characters at max so specify this in system prompt
    return (
        "You are a helpful assistant.\n"
        "Your output must always be less or equal to 4096 characters.\n"
    )


async def llm_chatting_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    llm_chat = context.user_data[LLM_CHAT_USER_DATA_FIELD]

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

    bot_message = await update.message.reply_text(strings.LLMChatting.THINKING)
    try:
        answer = await get_next_llm_chat_completion(llm_chat.data)
        llm_messages.append({"role": "assistant", "content": answer})
        await bot_message.edit_text(answer)
    except Exception as ex:
        await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)
        raise ex


async def chat_creation_cancellation_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Send a message when the command /cancel is issued."""
    await update.message.reply_text(strings.Global.CHAT_CREATION_CANCELLATION)
    return ConversationHandler.END


async def llm_mode_selection_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        strings.LLMModeSelection.UNAVAILABLE_COMMAND_FORMAT.format("start")
    )


async def llm_mode_selection_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        strings.LLMModeSelection.UNAVAILABLE_COMMAND_FORMAT.format("new")
    )


async def llm_verbosity_selection_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        strings.LLMVerbositySelection.UNAVAILABLE_COMMAND_FORMAT.format("start")
    )


async def llm_verbosity_selection_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        strings.LLMVerbositySelection.UNAVAILABLE_COMMAND_FORMAT.format("new")
    )


def generate_help(data: dict[str, str]) -> str:
    rows = []
    for command, description in data.items():
        rows.append(f"/{command} - {description}.")
    return "\n".join(rows)


async def chat_creation_help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued while configuring LLM."""
    await update.message.reply_text(
        generate_help(
            {"help": "Вывести это сообщение", "cancel": "Отменить создание чата"}
        )
    )


async def llm_chatting_unsupported_command_message(
    command: str, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.message.reply_text(
        f"Команда /{command} недоступна во время чата с LLM.\n"
        "Чтобы завершить чат с LLM используйте команду /stop."
    )


async def llm_chatting_stop_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Send a message when the command /stop is issued."""
    await update.message.reply_text("Чат завершён.")
    context.user_data[LLM_CHAT_USER_DATA_FIELD].settings = None
    context.user_data[LLM_CHAT_USER_DATA_FIELD].data = None
    return ConversationHandler.END


async def llm_chatting_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /start is issued while chatting with LLM."""
    await llm_chatting_unsupported_command_message("start", update, context)


async def llm_chatting_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /new is issued while configuring LLM."""
    await llm_chatting_unsupported_command_message("new", update, context)


async def llm_chatting_help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued while chatting with LLM."""
    await update.message.reply_text(
        generate_help({"help": "Вывести это сообщение", "stop": "Завершить чат с LLM"})
    )


async def entry_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> ConversationState:
    """Send a message when the command /new is issued."""
    context.user_data[LLM_CHAT_USER_DATA_FIELD] = llm.LLMChat(
        llm.LLMSettings(None, None), {}
    )
    await llm_mode_selection_entry(update, context)
    return ConversationState.LLM_MODE_SELECTION


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
            ConversationState.LLM_MODE_SELECTION: [
                CallbackQueryHandler(llm_mode_selection_handler),
                CommandHandler("cancel", chat_creation_cancellation_handler),
                CommandHandler("start", llm_mode_selection_start_handler),
                CommandHandler("new", llm_mode_selection_new_handler),
                CommandHandler("help", chat_creation_help_handler),
            ],
            ConversationState.LLM_VERBOSITY_SELECTION: [
                CallbackQueryHandler(llm_verbosity_selection_handler),
                CommandHandler("cancel", chat_creation_cancellation_handler),
                CommandHandler("start", llm_verbosity_selection_start_handler),
                CommandHandler("new", llm_verbosity_selection_new_handler),
                CommandHandler("help", chat_creation_help_handler),
            ],
            ConversationState.LLM_CHATTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, llm_chatting_handler),
                CommandHandler("stop", llm_chatting_stop_handler),
                CommandHandler("start", llm_chatting_start_handler),
                CommandHandler("new", llm_chatting_new_handler),
                CommandHandler("help", llm_chatting_help_handler),
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

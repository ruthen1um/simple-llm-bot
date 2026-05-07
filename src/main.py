"""
Simple and easy to deploy telegram bot for communication with LLM.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any
from warnings import filterwarnings

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    PicklePersistence,
    filters,
)
from telegram.warnings import PTBUserWarning

from quickllmbot import llm, strings


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


CONFIG: dict[str, str]

BOT_TOKEN: str
BOT_PERSISTENCE_FILE: str
LLM_API_URL: str

llm_client: httpx.AsyncClient
logger: logging.Logger


def check_env() -> None:
    """Checks required variables existence in environment."""
    if "BOT_TOKEN" not in os.environ:
        raise RuntimeError("BOT_TOKEN environment variable is not set")

    if "BOT_PERSISTENCE_FILE" not in os.environ:
        raise RuntimeError("BOT_PERSISTENCE_FILE environment variable is not set")

    if "LLM_API_URL" not in os.environ:
        raise RuntimeError("LLM_API_URL environment variable is not set")


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
            raise ValueError("s does not correspond to a valid LLMMode object")


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
            raise ValueError("s does not correspond to a valid LLMVerbosity object")


async def llm_mode_selection_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Sends message upon entering LLM_MODE_SELECTION conversation state."""
    message = update.message
    if message is None:
        raise TypeError("message is None")

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
    await message.reply_text(
        strings.LLMModeSelection.REQUEST, reply_markup=reply_markup
    )


async def llm_verbosity_selection_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Sends message upon entering LLM_VERBOSITY_SELECTION conversation state."""
    user = update.effective_user
    if user is None:
        raise TypeError("user is None")

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
    if user is None:
        raise TypeError("user is None")

    await user.send_message(strings.ChatCreation.SUCCESS)


async def llm_mode_selection_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> ConversationState:
    if context.user_data is None:
        context.user_data = {}

    query = update.callback_query
    if query is None:
        raise TypeError("query is None")

    data = query.data
    if data is None:
        raise TypeError("data is None")

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
    if context.user_data is None:
        context.user_data = {}

    query = update.callback_query
    if query is None:
        raise TypeError("query is None")

    data = query.data
    if data is None:
        raise TypeError("data is None")

    context.user_data[LLM_CHAT_USER_DATA_FIELD].settings.verbosity = get_llm_verbosity(
        data
    )

    await query.answer()
    await query.edit_message_text(
        strings.LLMVerbositySelection.SELECTION_FORMAT.format(verbosity=data)
    )

    await llm_chatting_entry(update, context)
    return ConversationState.LLM_CHATTING


async def get_next_llm_chat_completion(data: dict[str, Any]) -> str:
    response = await llm_client.post(
        CHAT_COMPLETIONS_API_PATH,
        json=data,
    )
    response.raise_for_status()
    response_data = response.json()

    choices = response_data["choices"]
    if len(choices) > 1:
        raise RuntimeError("LLM returned more than one choice")

    if choices[0]["finish_reason"] != "stop":
        raise RuntimeError("LLM finish reason is not `stop`")

    completion = choices[0]["message"]["content"]
    if not isinstance(completion, str):
        raise TypeError("completion type is not str")

    return completion


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
    if context.user_data is None:
        context.user_data = {}

    message = update.message
    if message is None:
        raise TypeError("message is None")

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

    prompt = message.text
    llm_messages = llm_chat.data["messages"]
    llm_messages.append({"role": "user", "content": prompt})

    bot_message = await message.reply_text(strings.LLMChatting.THINKING)
    try:
        answer = await get_next_llm_chat_completion(llm_chat.data)
        llm_messages.append({"role": "assistant", "content": answer})
        await bot_message.edit_text(answer)
    except Exception:
        await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)
        raise


async def chat_creation_cancellation_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Send a message when the command /cancel is issued."""
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(strings.ChatCreation.CANCELLATION)
    return ConversationHandler.END


async def llm_mode_selection_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        strings.LLMModeSelection.UNAVAILABLE_COMMAND_FORMAT.format(command="start")
    )


async def llm_mode_selection_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        strings.LLMModeSelection.UNAVAILABLE_COMMAND_FORMAT.format(command="new")
    )


async def llm_verbosity_selection_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        strings.LLMVerbositySelection.UNAVAILABLE_COMMAND_FORMAT.format(command="start")
    )


async def llm_verbosity_selection_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        strings.LLMVerbositySelection.UNAVAILABLE_COMMAND_FORMAT.format(command="new")
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
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        generate_help(
            {
                "help": strings.ChatCreation.HELP_COMMAND_DESCRIPTION,
                "cancel": strings.ChatCreation.CANCEL_COMMAND_DESCRIPTION,
            }
        )
    )


async def llm_chatting_unsupported_command_message(
    command: str, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        strings.LLMChatting.UNAVAILABLE_COMMAND_FORMAT.format(command=command)
    )


async def llm_chatting_stop_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Send a message when the command /stop is issued."""
    if context.user_data is None:
        context.user_data = {}

    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(strings.LLMChatting.CHAT_FINISHED)

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
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        generate_help(
            {
                "help": strings.LLMChatting.HELP_COMMAND_DESCRIPTION,
                "stop": strings.LLMChatting.STOP_COMMAND_DESCRIPTION,
            }
        )
    )


async def entry_new_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> ConversationState:
    """Send a message when the command /new is issued."""
    if context.user_data is None:
        context.user_data = {}

    context.user_data[LLM_CHAT_USER_DATA_FIELD] = llm.LLMChat(
        llm.LLMSettings(None, None), {}
    )

    await llm_mode_selection_entry(update, context)
    return ConversationState.LLM_MODE_SELECTION


async def global_start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /start is issued."""
    message = update.message
    if message is None:
        raise TypeError("message is None")

    user = update.effective_user
    if user is None:
        raise TypeError("user is None")

    await message.reply_text(strings.Global.START_FORMAT.format(name=user.first_name))


async def global_help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued when not in a
    conversation."""
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(
        generate_help(
            {
                "start": strings.Global.START_COMMAND_DESCRIPTION,
                "help": strings.Global.HELP_COMMAND_DESCRIPTION,
                "new": strings.Global.NEW_COMMAND_DESCRIPTION,
            }
        )
    )


async def global_unknown_command_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when an unknown command is issued."""
    message = update.message
    if message is None:
        raise TypeError("message is None")

    await message.reply_text(strings.Global.UNKNOWN_COMMAND)


def main() -> None:
    """Start the bot."""
    persistence = PicklePersistence(filepath=BOT_PERSISTENCE_FILE)
    application = (
        Application.builder().token(BOT_TOKEN).persistence(persistence).build()
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
    check_env()
    BOT_TOKEN = os.environ["BOT_TOKEN"]
    BOT_PERSISTENCE_FILE = os.environ["BOT_PERSISTENCE_FILE"]
    LLM_API_URL = os.environ["LLM_API_URL"]

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

    # Disable python-telegram-bot warning
    filterwarnings(
        action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
    )

    main()

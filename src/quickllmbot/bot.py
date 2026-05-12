import logging
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any
from warnings import filterwarnings

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
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

from . import llm, strings


class ConversationState(Enum):
    """Represents the conversation state."""

    LLM_MODE_SELECTION = 1
    LLM_VERBOSITY_SELECTION = 2
    LLM_CHATTING = 3


CHAT_COMPLETIONS_API_PATH = "/v1/chat/completions"

LLM_CHAT_USER_DATA_FIELD = "llm_chat"


class CallbackData:
    LLM_MODE_TEXT = "text"
    LLM_MODE_DOCUMENTS = "documents"
    LLM_MODE_IMAGES = "images"

    LLM_VERBOSITY_SHORT = "short"
    LLM_VERBOSITY_DEFAULT = "default"
    LLM_VERBOSITY_VERBOSE = "verbose"


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


async def get_next_llm_chat_completion(
    client: httpx.AsyncClient, data: dict[str, Any]
) -> str:
    response = await client.post(
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


def generate_help(data: dict[str, str]) -> str:
    rows = []
    for command, description in data.items():
        rows.append(f"/{command} - {description}.")
    return "\n".join(rows)


def get_system_prompt(settings: llm.LLMSettings) -> str:
    """Returns system prompt corresponding to provided settings."""
    # TODO: implement system prompt selection logic
    return "You are a helpful assistant."


class QuickLLMBot:
    def __init__(
        self,
        logger: logging.Logger,
        bot_token: str,
        bot_persistence_file_path: Path,
        inference_response_timeout: int,
        inference_api_url: str,
        inference_api_key: str | None,
    ):
        # TODO: add simple messages handler in all contexts
        # Disable useless PTB warning
        filterwarnings(
            action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        self._logger = logger

        persistence = PicklePersistence(filepath=bot_persistence_file_path)
        self._application = (
            Application.builder().token(bot_token).persistence(persistence).build()
        )

        self._register_handlers()

        headers = {}
        if inference_api_key:
            headers["Authorization"] = f"Bearer {inference_api_key}"

        self._inference_api_client = httpx.AsyncClient(
            base_url=inference_api_url,
            headers=headers,
            timeout=httpx.Timeout(inference_response_timeout),
        )

    def run(self):
        try:
            self._application.run_polling(allowed_updates=Update.ALL_TYPES)
        except TelegramError as e:
            self._logger.error("Telegram error: %s", e.message)

    class _Handlers:
        def __init__(self, bot: QuickLLMBot):
            self.bot = bot
            self.glob = self.Global(bot)
            self.chat_creation = self.ChatCreation(bot)
            self.llm_chatting = self.LLMChatting(bot)
            self.llm_mode_selection = self.LLMModeSelection(bot)
            self.llm_verbosity_selection = self.LLMVerbositySelection(bot)

        class LLMModeSelection:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def callback_query(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> ConversationState:
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                query = update.callback_query
                if query is None:
                    raise TypeError("query is None")

                data = query.data
                if data is None:
                    raise TypeError("data is None")

                user_data[LLM_CHAT_USER_DATA_FIELD].settings.mode = get_llm_mode(data)

                await query.answer()
                await query.edit_message_text(
                    strings.LLMModeSelection.SELECTION_FORMAT.format(mode=data)
                )

                await self.bot._helpers.llm_verbosity_selection.entry(update, context)
                return ConversationState.LLM_VERBOSITY_SELECTION

            async def command_start(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(
                    strings.LLMModeSelection.UNAVAILABLE_COMMAND_FORMAT.format(
                        command="start"
                    )
                )

            async def command_new(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(
                    strings.LLMModeSelection.UNAVAILABLE_COMMAND_FORMAT.format(
                        command="new"
                    )
                )

        class LLMVerbositySelection:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def callback_query(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> ConversationState:
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                query = update.callback_query
                if query is None:
                    raise TypeError("query is None")

                data = query.data
                if data is None:
                    raise TypeError("data is None")

                user_data[
                    LLM_CHAT_USER_DATA_FIELD
                ].settings.verbosity = get_llm_verbosity(data)

                await query.answer()
                await query.edit_message_text(
                    strings.LLMVerbositySelection.SELECTION_FORMAT.format(
                        verbosity=data
                    )
                )

                await self.bot._helpers.llm_chatting.entry(update, context)
                return ConversationState.LLM_CHATTING

            async def command_start(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(
                    strings.LLMVerbositySelection.UNAVAILABLE_COMMAND_FORMAT.format(
                        command="start"
                    )
                )

            async def command_new(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(
                    strings.LLMVerbositySelection.UNAVAILABLE_COMMAND_FORMAT.format(
                        command="new"
                    )
                )

        class LLMChatting:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def command_stop(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> int:
                """Send a message when the command /stop is issued."""
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(strings.LLMChatting.CHAT_FINISHED)

                user_data[LLM_CHAT_USER_DATA_FIELD].settings = None
                user_data[LLM_CHAT_USER_DATA_FIELD].data = None

                return ConversationHandler.END

            async def command_start(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Send a message when the command /start is issued while chatting with LLM."""
                await self.bot._helpers.llm_chatting.unsupported_command_message(
                    "start", update, context
                )

            async def command_new(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Send a message when the command /new is issued while configuring LLM."""
                await self.bot._helpers.llm_chatting.unsupported_command_message(
                    "new", update, context
                )

            async def command_help(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
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

            async def text(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                chat = update.effective_chat
                if chat is None:
                    raise TypeError("chat is None")

                llm_chat = user_data[LLM_CHAT_USER_DATA_FIELD]

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
                if prompt is None:
                    raise TypeError("prompt is None")

                llm_messages = llm_chat.data["messages"]
                llm_messages.append({"role": "user", "content": prompt})

                bot_message = await message.reply_text(strings.LLMChatting.THINKING)
                try:
                    answer = await get_next_llm_chat_completion(
                        self.bot._inference_api_client, llm_chat.data
                    )
                    llm_messages.append({"role": "assistant", "content": answer})
                    parts = textwrap.wrap(answer, 4096)
                    for i, part in enumerate(parts):
                        if i == 0:
                            await bot_message.edit_text(part)
                        else:
                            await context.bot.send_message(
                                chat_id=chat.id,
                                text=part,
                            )
                except httpx.TimeoutException:
                    self.bot._logger.error("Connection to inference server timed out")
                    await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)
                except httpx.NetworkError:
                    self.bot._logger.error(
                        "Network error during communication with inference server"
                    )
                    await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)
                except httpx.HTTPStatusError as e:
                    self.bot._logger.error(
                        "Inference server returned error: %s %s",
                        e.response.status_code,
                        e.response.text,
                    )
                    await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)

        class Global:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def command_new(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> ConversationState:
                """Send a message when the command /new is issued."""
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                user_data[LLM_CHAT_USER_DATA_FIELD] = llm.LLMChat(
                    llm.LLMSettings(None, None), {}
                )

                await self.bot._helpers.llm_mode_selection.entry(update, context)
                return ConversationState.LLM_MODE_SELECTION

            async def command_start(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Send a message when the command /start is issued."""
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                user = update.effective_user
                if user is None:
                    raise TypeError("user is None")

                await message.reply_text(
                    strings.Global.START_FORMAT.format(name=user.first_name)
                )

            async def command_help(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
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

            async def unknown_command(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Send a message when an unknown command is issued."""
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(strings.Global.UNKNOWN_COMMAND)

        class ChatCreation:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def command_cancel(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> int:
                """Send a message when the command /cancel is issued."""
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(strings.ChatCreation.CANCELLATION)
                return ConversationHandler.END

            async def command_help(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
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

    class _Helpers:
        def __init__(self, bot: QuickLLMBot):
            self.bot = bot
            self.llm_chatting = self.LLMChatting(bot)
            self.llm_mode_selection = self.LLMModeSelection(bot)
            self.llm_verbosity_selection = self.LLMVerbositySelection(bot)

        class LLMModeSelection:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def entry(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
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
                # TODO: use send_message rather than reply_text
                await message.reply_text(
                    strings.LLMModeSelection.REQUEST, reply_markup=reply_markup
                )

        class LLMVerbositySelection:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def entry(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Sends message upon entering LLM_VERBOSITY_SELECTION conversation state."""
                chat = update.effective_chat
                if chat is None:
                    raise TypeError("chat is None")

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
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=strings.LLMVerbositySelection.REQUEST,
                    reply_markup=reply_markup,
                )

        class LLMChatting:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def entry(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                chat = update.effective_chat
                if chat is None:
                    raise TypeError("chat is None")

                await context.bot.send_message(
                    chat_id=chat.id,
                    text=strings.ChatCreation.SUCCESS,
                )

            async def unsupported_command_message(
                self, command: str, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(
                    strings.LLMChatting.UNAVAILABLE_COMMAND_FORMAT.format(
                        command=command
                    )
                )

    def _register_handlers(self) -> None:
        self._handlers = self._Handlers(self)
        self._helpers = self._Helpers(self)
        self._application.add_handler(
            ConversationHandler(
                entry_points=[CommandHandler("new", self._handlers.glob.command_new)],
                states={
                    ConversationState.LLM_MODE_SELECTION: [
                        CallbackQueryHandler(
                            self._handlers.llm_mode_selection.callback_query
                        ),
                        CommandHandler(
                            "cancel", self._handlers.chat_creation.command_cancel
                        ),
                        CommandHandler(
                            "start", self._handlers.llm_mode_selection.command_start
                        ),
                        CommandHandler(
                            "new", self._handlers.llm_mode_selection.command_new
                        ),
                        CommandHandler(
                            "help", self._handlers.chat_creation.command_help
                        ),
                    ],
                    ConversationState.LLM_VERBOSITY_SELECTION: [
                        CallbackQueryHandler(
                            self._handlers.llm_verbosity_selection.callback_query
                        ),
                        CommandHandler(
                            "cancel", self._handlers.chat_creation.command_cancel
                        ),
                        CommandHandler(
                            "start",
                            self._handlers.llm_verbosity_selection.command_start,
                        ),
                        CommandHandler(
                            "new", self._handlers.llm_verbosity_selection.command_new
                        ),
                        CommandHandler(
                            "help", self._handlers.chat_creation.command_help
                        ),
                    ],
                    ConversationState.LLM_CHATTING: [
                        CommandHandler(
                            "stop", self._handlers.llm_chatting.command_stop
                        ),
                        CommandHandler(
                            "start", self._handlers.llm_chatting.command_start
                        ),
                        CommandHandler("new", self._handlers.llm_chatting.command_new),
                        CommandHandler(
                            "help", self._handlers.llm_chatting.command_help
                        ),
                        MessageHandler(
                            filters.TEXT & ~filters.COMMAND,
                            self._handlers.llm_chatting.text,
                        ),
                    ],
                },
                fallbacks=[],
                name="main_conversation_handler",
                persistent=True,
            )
        )
        self._application.add_handler(
            CommandHandler("start", self._handlers.glob.command_start)
        )
        self._application.add_handler(
            CommandHandler("help", self._handlers.glob.command_help)
        )
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._handlers.glob.unknown_command)
        )
        # self._application.add_handler(
        #     MessageHandler(filters.TEXT, self._handlers.glob.text)
        # )

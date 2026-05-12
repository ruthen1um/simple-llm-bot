"""Telegram bot implementation for LLM inference with conversation state management."""

import logging
# import textwrap
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
from pdf_oxide import PdfDocument
from telegramify_markdown import convert

from . import llm, strings


class ConversationState(Enum):
    """Defines the possible states of a user conversation."""

    LLM_MODE_SELECTION = 1
    LLM_VERBOSITY_SELECTION = 2
    LLM_CHATTING = 3


CHAT_COMPLETIONS_API_PATH = "/v1/chat/completions"

LLM_CHAT_USER_DATA_FIELD = "llm_chat"

SUPPORTED_DOCUMENT_MIME_TYPES = {
    "text/plain",
    "application/pdf",
}

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
}


class CallbackData:
    LLM_MODE_TEXT = "text"
    LLM_MODE_DOCUMENTS = "documents"
    LLM_MODE_IMAGES = "images"

    LLM_VERBOSITY_SHORT = "short"
    LLM_VERBOSITY_DEFAULT = "default"
    LLM_VERBOSITY_VERBOSE = "verbose"


def get_llm_mode(s: str) -> llm.LLMMode:
    """Converts a callback string into the corresponding LLMMode enum."""

    if s is None:
        raise TypeError("s is None")

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
    """Converts a callback string into the corresponding LLMVerbosity enum."""

    if s is None:
        raise TypeError("s is None")

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
    """Sends a chat completion request to the inference API and returns the response."""

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
    """Generates a help message from a dictionary of commands and descriptions."""

    rows = []
    for command, description in data.items():
        rows.append(f"/{command} - {description}.")
    return "\n".join(rows)


def get_initial_chat_data(settings: llm.LLMSettings) -> dict:
    """Returns initial chat data structure with system prompt."""

    return {
        "temperature": 0.3,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": get_system_prompt(settings),
            }
        ],
    }


def get_system_prompt(settings: llm.LLMSettings) -> str:
    """Generates the system instruction prompt based on user-selected settings."""

    mode = settings.mode
    verbosity = settings.verbosity

    if mode is None or verbosity is None:
        raise ValueError("Both mode and verbosity must be set")

    mode_prompts = {
        llm.LLMMode.TEXT: "You are a helpful assistant for text-based tasks.",
        llm.LLMMode.DOCUMENTS: "You are a helpful assistant for document analysis and processing.",
        llm.LLMMode.IMAGES: "You are a helpful assistant for image-related tasks.",
    }

    verbosity_modifiers = {
        llm.LLMVerbosity.SHORT: "Keep your responses concise and to the point.",
        llm.LLMVerbosity.DEFAULT: "Provide balanced, informative responses.",
        llm.LLMVerbosity.VERBOSE: "Provide detailed, comprehensive explanations with examples.",
    }

    base_prompt = mode_prompts[mode]
    modifier = verbosity_modifiers[verbosity]

    return base_prompt + " " + modifier


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
        """Initialize the QuickLLMBot with required dependencies and configuration."""

        # Disable useless PTB warning
        filterwarnings(
            action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("pdf_oxide").setLevel(logging.WARNING)

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
        """Starts the bot's polling loop for Telegram updates."""

        try:
            self._application.run_polling(allowed_updates=Update.ALL_TYPES)
        except TelegramError as e:
            self._logger.error("Telegram error: %s", e.message)

    def _register_handlers(self) -> None:
        """Registers all conversation and command handlers with the Telegram application."""

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
                        MessageHandler(
                            filters.TEXT & ~filters.COMMAND,
                            self._handlers.chat_creation.text,
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
                        MessageHandler(
                            filters.TEXT & ~filters.COMMAND,
                            self._handlers.chat_creation.text,
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
                        MessageHandler(
                            filters.Document.ALL,
                            self._handlers.llm_chatting.file,
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
        self._application.add_handler(
            MessageHandler(filters.TEXT, self._handlers.glob.text)
        )

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
                """Handles inline keyboard button presses for LLM mode selection."""

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
                """Handles /start command during mode selection state."""

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
                """Handles /new command during mode selection state."""

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
                """Handles inline keyboard button presses for LLM verbosity selection."""

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
                """Handles /start command during verbosity selection state."""

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
                """Handles /new command during verbosity selection state."""

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
                """Ends the current chat session and clears user session data."""

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
                """Notifies the user that /start is unavailable while a chat is active."""

                await self.bot._helpers.llm_chatting.unsupported_command_message(
                    "start", update, context
                )

            async def command_new(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Notifies the user that /new is unavailable while a chat is active."""

                await self.bot._helpers.llm_chatting.unsupported_command_message(
                    "new", update, context
                )

            async def command_help(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Displays available commands for the active chat state."""

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

            async def file(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                file = message.document
                if file is None:
                    raise TypeError("file is None")

                llm_chat = user_data[LLM_CHAT_USER_DATA_FIELD]
                match llm_chat.settings.mode:
                    case llm.LLMMode.TEXT:
                        await message.reply_text(
                            strings.LLMChatting.TEXT_FILES_NOT_ALLOWED
                        )
                    case llm.LLMMode.DOCUMENTS:
                        await self.document(update, context)
                    case llm.LLMMode.IMAGES:
                        await self.image(update, context)

            async def document(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                document = message.document
                if document is None:
                    raise TypeError("document is None")

                mime_type = document.mime_type
                if mime_type is None or mime_type not in SUPPORTED_DOCUMENT_MIME_TYPES:
                    await message.reply_text(
                        strings.LLMChatting.DOCUMENT_MIME_TYPE_NOT_ALLOWED
                    )
                    return

                processing_msg = await message.reply_text(
                    strings.LLMChatting.PROCESSING_DOCUMENT
                )

                file = await document.get_file()
                data = await file.download_as_bytearray()

                extracted_text: str | None = None
                if mime_type == "application/pdf":
                    try:
                        extracted_text = PdfDocument.from_bytes(
                            bytes(data)
                        ).to_markdown_all()
                    except Exception:
                        await processing_msg.edit_text(
                            strings.LLMChatting.DOCUMENT_PROCESSING_ERROR
                        )
                        return
                elif mime_type == "text/plain":
                    extracted_text = data.decode()

                llm_chat = user_data[LLM_CHAT_USER_DATA_FIELD]
                if not llm_chat.data:
                    llm_chat.data = get_initial_chat_data(llm_chat.settings)

                filename = document.file_name or "<UNNAMED DOCUMENT>"
                context_message = f"DOCUMENT: {filename}\n\n{extracted_text}"

                llm_chat.data["messages"].append(
                    {"role": "user", "content": context_message}
                )

                await processing_msg.edit_text(strings.LLMChatting.DOCUMENT_READY)

            async def image(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                image = message.document
                if image is None:
                    raise TypeError("image is None")

                mime_type = image.mime_type
                if mime_type is None or mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                    await message.reply_text(
                        strings.LLMChatting.IMAGE_MIME_TYPE_NOT_ALLOWED
                    )
                    return

            async def text(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Handles user text messages during active LLM chat."""

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
                    llm_chat.data = get_initial_chat_data(llm_chat.settings)

                llm_messages = llm_chat.data["messages"]

                prompt = message.text
                if prompt is None:
                    raise TypeError("prompt is None")

                llm_messages.append({"role": "user", "content": prompt})
                self.bot._logger.info("%s", llm_chat)

                bot_message = await message.reply_text(strings.LLMChatting.THINKING)
                try:
                    answer = await get_next_llm_chat_completion(
                        self.bot._inference_api_client, llm_chat.data
                    )
                    text, entities = convert(answer)
                    llm_messages.append({"role": "assistant", "content": text})
                    await bot_message.edit_text(text, entities=[e.to_dict() for e in entities])
                    # parts = textwrap.wrap(answer, 4096)
                    # for i, part in enumerate(parts):
                    #     if i == 0:
                    #         await bot_message.edit_text(part)
                    #     else:
                    #         await context.bot.send_message(
                    #             chat_id=chat.id,
                    #             text=part,
                    #         )
                except httpx.TimeoutException:
                    self.bot._logger.error("Connection to inference server timed out")
                    del llm_messages[-1]
                    await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)
                except httpx.NetworkError:
                    self.bot._logger.error(
                        "Network error during communication with inference server"
                    )
                    del llm_messages[-1]
                    await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)
                except httpx.HTTPStatusError as e:
                    self.bot._logger.error(
                        "Inference server returned error: %s %s",
                        e.response.status_code,
                        e.response.text,
                    )
                    del llm_messages[-1]
                    await bot_message.edit_text(strings.LLMChatting.COMMUNICATION_ERROR)

        class Global:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def command_new(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> ConversationState:
                """Initiates a new chat configuration sequence."""

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
                """Sends the initial greeting message to the user."""

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
                """Displays the general help message for the bot."""

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
                """Handles unrecognized commands by sending a default error message."""

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(strings.Global.UNKNOWN_COMMAND)

            async def text(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Handles user text messages when not in a conversation."""

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await self.command_start(update, context)

        class ChatCreation:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def command_cancel(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> int:
                """Aborts the chat creation process and returns to the idle state."""

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(strings.ChatCreation.CANCELLATION)
                return ConversationHandler.END

            async def command_help(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Displays help information specifically for the chat configuration phase."""

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

            async def text(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Resends the inline keyboard based on current conversation state."""

                user_data = context.user_data
                if not isinstance(user_data, dict):
                    raise TypeError("user_data is not a dict")

                llm_chat = user_data[LLM_CHAT_USER_DATA_FIELD]

                if llm_chat.settings.mode is None:
                    # In LLM_MODE_SELECTION state
                    await self.bot._helpers.llm_mode_selection.entry(update, context)
                elif llm_chat.settings.verbosity is None:
                    # In LLM_VERBOSITY_SELECTION state
                    await self.bot._helpers.llm_verbosity_selection.entry(
                        update, context
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
                """Displays the mode selection menu to the user."""

                chat = update.effective_chat
                if chat is None:
                    raise TypeError("chat is None")

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
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=strings.LLMModeSelection.REQUEST,
                    reply_markup=reply_markup,
                )

        class LLMVerbositySelection:
            def __init__(self, bot: QuickLLMBot):
                self.bot = bot

            async def entry(
                self, update: Update, context: ContextTypes.DEFAULT_TYPE
            ) -> None:
                """Displays the verbosity selection menu to the user."""

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
                """Sends confirmation message upon entering LLM_CHATTING state."""

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
                """Sends a message indicating a command is not supported in current state."""

                message = update.message
                if message is None:
                    raise TypeError("message is None")

                await message.reply_text(
                    strings.LLMChatting.UNAVAILABLE_COMMAND_FORMAT.format(
                        command=command
                    )
                )

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModeSelection:
    UNAVAILABLE_COMMAND_FORMAT = (
        "Команда /{command} недоступна во время выбора режима работы.\n"
        "Чтобы завершить создание чата используйте команду /cancel."
    )

    REQUEST = "Выберите режим работы:"
    SELECTION_FORMAT = "Выбранный режим работы: {mode}."

    TEXT_MODE_BUTTON = "Работа с текстом"
    DOCUMENTS_MODE_BUTTON = "Работа с документами"
    IMAGES_MODE_BUTTON = "Работа с изображениями"


@dataclass(frozen=True)
class LLMVerbositySelection:
    UNAVAILABLE_COMMAND_FORMAT = (
        "Команда /{command} недоступна во время выбора уровня подробности вывода.\n"
        "Чтобы завершить создание чата используйте команду /cancel."
    )

    REQUEST = "Выберите уровень подробности вывода:"
    SELECTION_FORMAT = "Выбранный уровень подробности вывода: {verbosity}."

    SHORT_VERBOSITY_BUTTON = "Краткий"
    DEFAULT_VERBOSITY_BUTTON = "Стандартный"
    VERBOSE_VERBOSITY_BUTTON = "Подробный"


@dataclass(frozen=True)
class LLMChatting:
    THINKING = "Думаю..."
    COMMUNICATION_ERROR = "Ошибка при взаимодействии с LLM."


@dataclass(frozen=True)
class Global:
    CHAT_CREATION_CANCELLATION = "Создание чата отменено."
    CHAT_CREATION_SUCCESS = (
        "Чат с LLM создан. Далее пишите ваши промпты.\n"
        "Для завершения чата используйте команду /stop."
    )

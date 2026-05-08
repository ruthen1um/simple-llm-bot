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


class LLMChatting:
    UNAVAILABLE_COMMAND_FORMAT = (
        "Команда /{command} недоступна во время чата с LLM.\n"
        "Чтобы завершить чат с LLM используйте команду /stop."
    )

    THINKING = "Думаю..."
    COMMUNICATION_ERROR = "Ошибка при взаимодействии с LLM."

    CHAT_FINISHED = "Чат завершён."

    HELP_COMMAND_DESCRIPTION = "Вывести это сообщение"
    STOP_COMMAND_DESCRIPTION = "Завершить чат с LLM"


class Global:
    UNKNOWN_COMMAND = (
        "Введена неизвестная команда.\n"
        "Используйте команду /help, чтобы посмотреть список доступных команд."
    )

    START_FORMAT = "Привет, {name}!\nИспользуйте команду /new чтобы начать новый чат."

    START_COMMAND_DESCRIPTION = "Вывести первоначальное сообщение"
    HELP_COMMAND_DESCRIPTION = "Вывести это сообщение"
    NEW_COMMAND_DESCRIPTION = "Начать новый чат"


class ChatCreation:
    CANCELLATION = "Создание чата отменено."
    SUCCESS = (
        "Чат с LLM создан. Далее пишите ваши промпты.\n"
        "Для завершения чата используйте команду /stop."
    )

    HELP_COMMAND_DESCRIPTION = "Вывести это сообщение"
    CANCEL_COMMAND_DESCRIPTION = "Отменить создание чата"

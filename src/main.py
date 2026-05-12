"""Entry point for the Telegram LLM bot application."""

import logging
import os
from pathlib import Path

from quickllmbot import QuickLLMBot


def check_env() -> None:
    """Validates that all required environment variables are set."""

    if "BOT_TOKEN" not in os.environ:
        raise RuntimeError("BOT_TOKEN environment variable is not set")

    if "BOT_DATA_DIR" not in os.environ:
        raise RuntimeError("BOT_DATA_DIR environment variable is not set")

    if "BOT_PERSISTENCE_FILE" not in os.environ:
        raise RuntimeError("BOT_PERSISTENCE_FILE environment variable is not set")

    if "INFERENCE_RESPONSE_TIMEOUT" not in os.environ:
        raise RuntimeError("INFERENCE_RESPONSE_TIMEOUT environment variable is not set")

    if "INFERENCE_API_URL" not in os.environ:
        raise RuntimeError("INFERENCE_API_URL environment variable is not set")


def mask_secret(secret: str, visible: int) -> str:
    """Returns a masked version of a sensitive string, showing only the last N characters."""

    if len(secret) <= visible:
        return "*" * len(secret)
    return "*" * (len(secret) - visible) + secret[-visible:]


if __name__ == "__main__":
    check_env()

    BOT_TOKEN = os.environ["BOT_TOKEN"]
    BOT_DATA_DIR = Path(os.environ["BOT_DATA_DIR"])
    BOT_PERSISTENCE_FILE = Path(os.environ["BOT_PERSISTENCE_FILE"])
    INFERENCE_RESPONSE_TIMEOUT = int(os.environ["INFERENCE_RESPONSE_TIMEOUT"])
    INFERENCE_API_URL = os.environ["INFERENCE_API_URL"]
    # Required only for some servers, so purely optional
    INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY")

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger("QuickLLMBot")

    logger.info("BOT_TOKEN: %s", mask_secret(BOT_TOKEN, 4))
    logger.info("BOT_DATA_DIR: %s", BOT_DATA_DIR)
    logger.info("BOT_PERSISTENCE_FILE: %s", BOT_PERSISTENCE_FILE)
    logger.info("INFERENCE_RESPONSE_TIMEOUT: %d", INFERENCE_RESPONSE_TIMEOUT)
    logger.info("INFERENCE_API_URL: %s", INFERENCE_API_URL)
    logger.info(
        "INFERENCE_API_KEY: %s",
        "<not set>" if INFERENCE_API_KEY is None else mask_secret(INFERENCE_API_KEY, 4),
    )

    bot = QuickLLMBot(
        logger,
        BOT_TOKEN,
        BOT_DATA_DIR / BOT_PERSISTENCE_FILE,
        INFERENCE_RESPONSE_TIMEOUT,
        INFERENCE_API_URL,
        INFERENCE_API_KEY,
    )
    bot.run()

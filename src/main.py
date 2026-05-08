"""
Simple and easy-to-deploy Telegram bot for LLM inference.
"""

import os
import logging

from quickllmbot import QuickLLMBot


def check_env() -> None:
    """Checks required variables existence in environment."""
    if "BOT_TOKEN" not in os.environ:
        raise RuntimeError("BOT_TOKEN environment variable is not set")

    if "BOT_PERSISTENCE_FILE" not in os.environ:
        raise RuntimeError("BOT_PERSISTENCE_FILE environment variable is not set")

    if "INFERENCE_API_URL" not in os.environ:
        raise RuntimeError("INFERENCE_API_URL environment variable is not set")


if __name__ == "__main__":
    check_env()

    BOT_TOKEN = os.environ["BOT_TOKEN"]
    BOT_PERSISTENCE_FILE = os.environ["BOT_PERSISTENCE_FILE"]
    INFERENCE_API_URL = os.environ["INFERENCE_API_URL"]

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    bot = QuickLLMBot(BOT_TOKEN, BOT_PERSISTENCE_FILE, INFERENCE_API_URL)
    bot.run()

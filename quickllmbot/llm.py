from dataclasses import dataclass
from typing import Any
from enum import Enum


class LLMMode(Enum):
    """Represents mode of LLM operation."""

    TEXT = 1
    DOCUMENTS = 2
    IMAGES = 3


class LLMVerbosity(Enum):
    """Represents level of LLM verbosity."""

    SHORT = 1
    DEFAULT = 2
    VERBOSE = 3


@dataclass
class LLMSettings:
    """Stores LLM settings."""

    mode: LLMMode | None
    verbosity: LLMVerbosity | None


@dataclass
class LLMChat:
    """Stores LLM chat data and settings."""

    settings: LLMSettings | None
    data: dict[str, Any] | None

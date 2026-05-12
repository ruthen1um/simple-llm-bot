"""Data models and enums for LLM configuration and chat management."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class LLMMode(Enum):
    """Operational modes for the LLM."""

    TEXT = 1
    DOCUMENTS = 2
    IMAGES = 3


class LLMVerbosity(Enum):
    """Output detail levels for LLM responses."""

    SHORT = 1
    DEFAULT = 2
    VERBOSE = 3


@dataclass
class LLMSettings:
    """Container for user-defined LLM operational preferences."""

    mode: LLMMode | None
    verbosity: LLMVerbosity | None


@dataclass
class LLMChat:
    """Container for an active chat session's configuration and history."""

    settings: LLMSettings | None
    data: dict[str, Any] | None

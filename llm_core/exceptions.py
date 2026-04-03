from __future__ import annotations


class LLMCoreError(Exception):
    """Base exception for llm_core."""


class PromptNotFoundError(LLMCoreError):
    """Raised when a prompt template name does not exist in prompt.json."""


class PromptLanguageNotFoundError(LLMCoreError):
    """Raised when a prompt does not provide the requested language."""


class PromptRenderError(LLMCoreError):
    """Raised when prompt rendering fails."""


class LLMRequestError(LLMCoreError):
    """Raised when the SDK request to the LLM provider fails."""


class LLMEmptyResponseError(LLMCoreError):
    """Raised when the model returns no usable content."""


class LLMJsonDecodeError(LLMCoreError):
    """Raised when a final JSON response cannot be decoded."""


class JsonlStreamParseError(LLMCoreError):
    """Raised when a JSONL stream line cannot be decoded."""

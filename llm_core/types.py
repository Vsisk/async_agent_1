from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | dict[str, Any] | list[Any]


@dataclass(slots=True)
class PromptRenderResult:
    prompt_text: str
    used_templates: list[str]
    variables: dict[str, Any]


@dataclass(slots=True)
class LLMFinalResponse:
    parsed: dict[str, Any] | list[Any] | None
    raw_text: str
    prompt_text: str
    model: str | None
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class StreamJsonlObject:
    object: dict[str, Any] | list[Any]
    raw_line: str
    index: int


@dataclass(slots=True)
class JsonlParseEvent:
    parsed: dict[str, Any] | list[Any]
    raw_line: str


@dataclass(slots=True)
class PromptDefinition:
    zh: str | None = None
    en: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "PromptDefinition":
        return cls(
            zh=value.get("zh"),
            en=value.get("en"),
        )


@dataclass(slots=True)
class PromptCatalog:
    prompts: dict[str, PromptDefinition] = field(default_factory=dict)

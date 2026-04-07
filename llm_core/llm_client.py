from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Mapping
from pathlib import Path
from typing import Any, Literal, overload

try:
    from openai import AsyncOpenAI
    _OPENAI_SDK_AVAILABLE = True
except ModuleNotFoundError:
    AsyncOpenAI = Any  # type: ignore[assignment]
    _OPENAI_SDK_AVAILABLE = False

from llm_core.exceptions import (
    LLMEmptyResponseError,
    LLMJsonDecodeError,
    LLMRequestError,
)
from llm_core.jsonl_parser import IncrementalJsonlParser
from llm_core.prompt_manager import PromptManager
from llm_core.types import LLMFinalResponse, StreamJsonlObject


class OpenAILLMClient:
    """Unified business-facing LLM client with one entrypoint."""

    RESERVED_FIELDS = frozenset(
        {
            "prompt_template",
            "lang",
            "model",
            "temperature",
            "max_tokens",
            "timeout",
            "response_format",
            "stream",
            "strict",
            "image_base64",
        }
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        prompt_manager: PromptManager | None = None,
        prompt_file: str | Path | None = None,
        default_model: str = "gpt-4.1",
        default_lang: str = "zh",
        default_temperature: float = 0,
    ) -> None:
        if client is None and not _OPENAI_SDK_AVAILABLE:
            raise ModuleNotFoundError(
                "openai package is required to instantiate OpenAILLMClient. "
                "Install it with `pip install openai`."
            )
        self._client = client or AsyncOpenAI()
        self._prompt_manager = prompt_manager or PromptManager(
            prompt_file=prompt_file,
            default_lang=default_lang,
        )
        self._default_model = default_model
        self._default_lang = default_lang
        self._default_temperature = default_temperature

    @classmethod
    def split_request_arguments(
        cls,
        request_arguments: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split reserved generate_result_by_llm fields from prompt variables."""

        reserved_arguments: dict[str, Any] = {}
        prompt_variables: dict[str, Any] = {}
        for key, value in request_arguments.items():
            if key in cls.RESERVED_FIELDS:
                reserved_arguments[key] = value
            else:
                prompt_variables[key] = value
        return reserved_arguments, prompt_variables

    @overload
    def generate_result_by_llm(
        self,
        *,
        prompt_template: list[str],
        lang: str = "zh",
        response_format: dict[str, Any] | None = None,
        stream: Literal[False] = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        strict: bool = True,
        image_base64: str | None = None,
        **kwargs: Any,
    ) -> Awaitable[LLMFinalResponse]: ...

    @overload
    def generate_result_by_llm(
        self,
        *,
        prompt_template: list[str],
        lang: str = "zh",
        response_format: dict[str, Any] | None = None,
        stream: Literal[True],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        strict: bool = True,
        image_base64: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamJsonlObject]: ...

    def generate_result_by_llm(
        self,
        *,
        prompt_template: list[str],
        lang: str = "zh",
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        strict: bool = True,
        image_base64: str | None = None,
        **kwargs: Any,
    ) -> Awaitable[LLMFinalResponse] | AsyncIterator[StreamJsonlObject]:
        request_arguments = {
            "prompt_template": prompt_template,
            "lang": lang,
            "response_format": response_format,
            "stream": stream,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "strict": strict,
            "image_base64": image_base64,
            **kwargs,
        }
        _, prompt_variables = self.split_request_arguments(request_arguments)

        if stream:
            return self._generate_stream(
                prompt_template=prompt_template,
                lang=lang,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                strict=strict,
                response_format=response_format,
                prompt_variables=prompt_variables,
                image_base64=image_base64,
            )

        return self._generate_final(
            prompt_template=prompt_template,
            lang=lang,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            strict=strict,
            response_format=response_format,
            prompt_variables=prompt_variables,
            image_base64=image_base64,
        )

    async def _generate_final(
        self,
        *,
        prompt_template: list[str],
        lang: str,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        strict: bool,
        response_format: dict[str, Any] | None,
        prompt_variables: Mapping[str, Any],
        image_base64: str | None,
    ) -> LLMFinalResponse:
        prompt_text = self._render_prompt(
            prompt_template=prompt_template,
            lang=lang,
            strict=strict,
            prompt_variables=prompt_variables,
        )
        messages = self._build_messages(prompt_text, image_base64=image_base64)
        payload = self._build_chat_payload(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            messages=messages,
            stream=False,
        )

        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMRequestError(f"OpenAI request failed: {exc}") from exc

        raw_text = self._extract_message_text(response)
        if not raw_text.strip():
            raise LLMEmptyResponseError("LLM returned an empty response.")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMJsonDecodeError(
                f"Failed to decode final JSON response: {exc.msg}"
            ) from exc

        return LLMFinalResponse(
            parsed=parsed,
            raw_text=raw_text,
            prompt_text=prompt_text,
            model=getattr(response, "model", None),
            usage=self._normalize_usage(getattr(response, "usage", None)),
        )

    async def _generate_stream(
        self,
        *,
        prompt_template: list[str],
        lang: str,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        strict: bool,
        response_format: dict[str, Any] | None,
        prompt_variables: Mapping[str, Any],
        image_base64: str | None,
    ) -> AsyncIterator[StreamJsonlObject]:
        prompt_text = self._render_prompt(
            prompt_template=prompt_template,
            lang=lang,
            strict=strict,
            prompt_variables=prompt_variables,
        )
        messages = self._build_messages(prompt_text, image_base64=image_base64)
        payload = self._build_chat_payload(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            messages=messages,
            stream=True,
        )
        parser = IncrementalJsonlParser()
        index = 0

        try:
            stream_response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMRequestError(f"OpenAI stream request failed: {exc}") from exc

        async for chunk in stream_response:
            delta_text = self._extract_stream_delta_text(chunk)
            if not delta_text:
                continue

            for event in parser.feed(delta_text):
                yield StreamJsonlObject(
                    object=event.parsed,
                    raw_line=event.raw_line,
                    index=index,
                )
                index += 1

        for event in parser.flush():
            yield StreamJsonlObject(
                object=event.parsed,
                raw_line=event.raw_line,
                index=index,
            )
            index += 1

    def _render_prompt(
        self,
        *,
        prompt_template: list[str],
        lang: str,
        strict: bool,
        prompt_variables: Mapping[str, Any],
    ) -> str:
        return self._prompt_manager.render(
            prompt_template=prompt_template,
            lang=lang or self._default_lang,
            variables=prompt_variables,
            strict=strict,
        ).prompt_text

    @staticmethod
    def _build_messages(
        prompt_text: str,
        *,
        image_base64: str | None = None,
    ) -> list[dict[str, Any]]:
        if image_base64:
            image_url = image_base64
            if not image_base64.startswith("data:"):
                image_url = f"data:image/png;base64,{image_base64}"
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

        return [{"role": "user", "content": prompt_text}]

    def _build_chat_payload(
        self,
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        response_format: dict[str, Any] | None,
        messages: list[dict[str, Any]],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "stream": stream,
        }
        resolved_temperature = (
            self._default_temperature if temperature is None else temperature
        )
        payload["temperature"] = resolved_temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if timeout is not None:
            payload["timeout"] = timeout
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    @staticmethod
    def _extract_message_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", None)
        return OpenAILLMClient._extract_content_text(content)

    @staticmethod
    def _extract_stream_delta_text(chunk: Any) -> str:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""

        content = getattr(delta, "content", None)
        return OpenAILLMClient._extract_content_text(content)

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        parts.append(part_text)
                    elif isinstance(part_text, dict):
                        value = part_text.get("value")
                        if value is not None:
                            parts.append(str(value))
                    continue

                maybe_text = getattr(part, "text", None)
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
                    continue

                maybe_value = getattr(maybe_text, "value", None)
                if maybe_value is not None:
                    parts.append(str(maybe_value))
            return "".join(parts)

        return str(content)

    @staticmethod
    def _normalize_usage(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return dict(usage.model_dump())
        if hasattr(usage, "__dict__"):
            return dict(vars(usage))
        return None

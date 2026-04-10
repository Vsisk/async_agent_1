from __future__ import annotations

import ast
import json
import logging
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

logger = logging.getLogger(__name__)


class OpenAILLMClient:
    """Unified business-facing LLM client with one entrypoint."""

    RESERVED_FIELDS = frozenset(
        {
            "prompt_template",
            "lang",
            "model",
            "temperature",
            "enable_thinking",
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
        enable_thinking: bool = False,
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
        enable_thinking: bool = False,
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
        enable_thinking: bool = False,
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
            "enable_thinking": enable_thinking,
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
                enable_thinking=enable_thinking,
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
            enable_thinking=enable_thinking,
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
        enable_thinking: bool,
        max_tokens: int | None,
        timeout: float | None,
        strict: bool,
        response_format: dict[str, Any] | None,
        prompt_variables: Mapping[str, Any],
        image_base64: str | None,
    ) -> LLMFinalResponse:
        logger.info(
            "Starting final LLM request: templates=%s lang=%s model=%s thinking=%s response_format=%s image=%s",
            prompt_template,
            lang,
            model or self._default_model,
            enable_thinking,
            response_format is not None,
            image_base64 is not None,
        )
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
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            messages=messages,
            stream=False,
        )

        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            logger.exception("Final LLM request failed: model=%s", payload["model"])
            raise LLMRequestError(f"OpenAI request failed: {exc}") from exc

        raw_text = self._extract_message_text(response)
        if not raw_text.strip():
            logger.warning("Final LLM response was empty: model=%s", getattr(response, "model", None))
            raise LLMEmptyResponseError("LLM returned an empty response.")
        normalized_text = self._sanitize_json_text(raw_text)
        logger.debug(
            "Final LLM response received: model=%s raw_length=%s normalized_length=%s preview=%s",
            getattr(response, "model", None),
            len(raw_text),
            len(normalized_text),
            self._preview_text(normalized_text),
        )

        try:
            parsed = self._decode_json_like_text(normalized_text)
        except ValueError as exc:
            logger.exception(
                "Failed to decode final JSON response: preview=%s",
                self._preview_text(normalized_text),
            )
            raise LLMJsonDecodeError(
                f"Failed to decode final JSON response: {exc}"
            ) from exc

        logger.info(
            "Completed final LLM request: model=%s usage=%s parsed_type=%s",
            getattr(response, "model", None),
            self._normalize_usage(getattr(response, "usage", None)),
            type(parsed).__name__,
        )
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
        enable_thinking: bool,
        max_tokens: int | None,
        timeout: float | None,
        strict: bool,
        response_format: dict[str, Any] | None,
        prompt_variables: Mapping[str, Any],
        image_base64: str | None,
    ) -> AsyncIterator[StreamJsonlObject]:
        logger.info(
            "Starting streaming LLM request: templates=%s lang=%s model=%s thinking=%s response_format=%s image=%s",
            prompt_template,
            lang,
            model or self._default_model,
            enable_thinking,
            response_format is not None,
            image_base64 is not None,
        )
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
            enable_thinking=enable_thinking,
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
            logger.exception("Streaming LLM request failed: model=%s", payload["model"])
            raise LLMRequestError(f"OpenAI stream request failed: {exc}") from exc

        async for chunk in stream_response:
            delta_text = self._extract_stream_delta_text(chunk)
            if not delta_text:
                continue
            logger.debug(
                "Received stream delta: model=%s delta_length=%s preview=%s",
                payload["model"],
                len(delta_text),
                self._preview_text(delta_text),
            )

            for event in parser.feed(delta_text):
                logger.debug(
                    "Yielding stream JSON object: index=%s raw_line=%s",
                    index,
                    self._preview_text(event.raw_line),
                )
                yield StreamJsonlObject(
                    object=event.parsed,
                    raw_line=event.raw_line,
                    index=index,
                )
                index += 1

        for event in parser.flush():
            logger.debug(
                "Yielding flushed stream JSON object: index=%s raw_line=%s",
                index,
                self._preview_text(event.raw_line),
            )
            yield StreamJsonlObject(
                object=event.parsed,
                raw_line=event.raw_line,
                index=index,
            )
            index += 1
        logger.info("Completed streaming LLM request: model=%s yielded=%s", payload["model"], index)

    def _render_prompt(
        self,
        *,
        prompt_template: list[str],
        lang: str,
        strict: bool,
        prompt_variables: Mapping[str, Any],
    ) -> str:
        logger.debug(
            "Rendering prompt via PromptManager: templates=%s lang=%s strict=%s variable_keys=%s",
            prompt_template,
            lang or self._default_lang,
            strict,
            sorted(prompt_variables.keys()),
        )
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
        enable_thinking: bool,
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
        reasoning_effort = self._resolve_reasoning_effort(
            model_name=payload["model"],
            enable_thinking=enable_thinking,
        )
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if timeout is not None:
            payload["timeout"] = timeout
        if response_format is not None:
            payload["response_format"] = response_format
        logger.debug(
            "Built chat payload: model=%s stream=%s temperature=%s max_tokens=%s timeout=%s reasoning_effort=%s response_format=%s",
            payload["model"],
            stream,
            payload["temperature"],
            payload.get("max_tokens"),
            payload.get("timeout"),
            payload.get("reasoning_effort"),
            response_format is not None,
        )
        return payload

    @staticmethod
    def _resolve_reasoning_effort(
        *,
        model_name: str,
        enable_thinking: bool,
    ) -> str | None:
        normalized_name = model_name.lower()

        if normalized_name.startswith("gpt-5.1"):
            return "medium" if enable_thinking else "none"

        reasoning_model_prefixes = ("gpt-5", "o1", "o3", "o4")
        if normalized_name.startswith(reasoning_model_prefixes):
            return "medium" if enable_thinking else None

        return None

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

    @staticmethod
    def _sanitize_json_text(raw_text: str) -> str:
        normalized_text = raw_text.strip()
        if not normalized_text.startswith("```"):
            return normalized_text
        if normalized_text.endswith("```"):
            inline_content = OpenAILLMClient._strip_inline_code_fence(normalized_text)
            if inline_content is not None:
                return inline_content

        lines = normalized_text.splitlines()
        if not lines:
            return normalized_text

        first_line = lines[0].strip().lower()
        if first_line == "```" or first_line.startswith("```json"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        sanitized = "\n".join(lines).strip()
        logger.debug(
            "Sanitized fenced JSON response: before_length=%s after_length=%s preview=%s",
            len(raw_text),
            len(sanitized),
            OpenAILLMClient._preview_text(sanitized),
        )
        return sanitized

    @staticmethod
    def _decode_json_like_text(text: str) -> dict[str, Any] | list[Any] | Any:
        json_error_message = ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as json_exc:
            json_error_message = json_exc.msg
            logger.debug(
                "Standard JSON decode failed, trying literal_eval fallback: preview=%s",
                OpenAILLMClient._preview_text(text),
            )

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as literal_exc:
            raise ValueError(json_error_message or str(literal_exc)) from literal_exc

        if not isinstance(parsed, (dict, list)):
            raise ValueError("Decoded content must be a JSON object or array.")

        logger.info(
            "Decoded response with literal_eval fallback: parsed_type=%s",
            type(parsed).__name__,
        )
        return parsed

    @staticmethod
    def _strip_inline_code_fence(candidate: str) -> str | None:
        if not candidate.startswith("```") or not candidate.endswith("```"):
            return None

        inner_content = candidate[3:-3].strip()
        if not inner_content:
            return ""

        lowered = inner_content.lower()
        if lowered.startswith("json"):
            inner_content = inner_content[4:].strip()

        return inner_content.strip()

    @staticmethod
    def _preview_text(value: str, max_length: int = 120) -> str:
        single_line = " ".join(value.split())
        if len(single_line) <= max_length:
            return single_line
        return single_line[: max_length - 3] + "..."

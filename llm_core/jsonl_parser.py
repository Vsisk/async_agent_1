from __future__ import annotations

import ast
import json
import logging
import re

from llm_core.exceptions import JsonlStreamParseError
from llm_core.types import JsonlParseEvent

logger = logging.getLogger(__name__)

_BARE_KEY_PATTERN = re.compile(
    r'(?P<prefix>[\{\[,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)(?P<suffix>\s*:)'
)


class IncrementalJsonlParser:
    """Incrementally parse structured JSON objects/arrays from arbitrary chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    @property
    def buffer(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> list[JsonlParseEvent]:
        logger.debug("Feeding structured JSON chunk: chunk_length=%s", len(chunk))
        self._buffer += chunk
        return self._drain_complete_values(include_tail=False)

    def flush(self) -> list[JsonlParseEvent]:
        logger.debug("Flushing structured JSON parser: buffer_length=%s", len(self._buffer))
        return self._drain_complete_values(include_tail=True)

    def _drain_complete_values(self, *, include_tail: bool) -> list[JsonlParseEvent]:
        events: list[JsonlParseEvent] = []

        while True:
            candidate, consumed = self._extract_next_candidate(include_incomplete=include_tail)
            if candidate is None:
                if consumed > 0:
                    self._buffer = self._buffer[consumed:]
                break

            self._buffer = self._buffer[consumed:]
            logger.debug(
                "Extracted structured JSON candidate: length=%s preview=%s",
                len(candidate),
                self._preview_text(candidate),
            )
            events.extend(self._parse_candidate(candidate))

        return events

    def _extract_next_candidate(self, *, include_incomplete: bool) -> tuple[str | None, int]:
        start_index = self._find_next_json_start()
        if start_index < 0:
            return None, len(self._buffer)

        depth = 0
        in_string = False
        escape = False
        quote_char = ""

        for index in range(start_index, len(self._buffer)):
            char = self._buffer[index]

            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == quote_char:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                quote_char = char
                continue

            if char in "{[":
                depth += 1
                continue

            if char in "}]":
                depth -= 1
                if depth < 0:
                    raise JsonlStreamParseError("Unexpected closing bracket while parsing stream.")
                if depth == 0:
                    return self._buffer[start_index : index + 1], index + 1

        if include_incomplete and self._buffer[start_index:].strip():
            raise JsonlStreamParseError("Incomplete JSON object or array at end of stream.")

        return None, start_index

    def _find_next_json_start(self) -> int:
        in_string = False
        escape = False
        quote_char = ""

        for index, char in enumerate(self._buffer):
            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == quote_char:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                quote_char = char
                continue

            if char in "{[":
                return index

        return -1

    def _parse_candidate(self, candidate: str) -> list[JsonlParseEvent]:
        sanitized = self._sanitize_candidate(candidate)
        if not sanitized:
            return []

        try:
            parsed = self._decode_json_like_text(sanitized)
        except ValueError as exc:
            logger.exception(
                "Failed to decode structured JSON candidate: preview=%s",
                self._preview_text(sanitized),
            )
            raise JsonlStreamParseError(
                f"Failed to decode JSON payload: {exc} (payload={sanitized!r})"
            ) from exc

        return self._expand_payload(parsed, raw_line=sanitized)

    @staticmethod
    def _sanitize_candidate(candidate: str) -> str:
        sanitized = candidate.strip()
        sanitized = re.sub(r"([\{\[])\s*,", r"\1", sanitized)
        sanitized = re.sub(r",\s*,+", ",", sanitized)
        sanitized = re.sub(r",\s*([\}\]])", r"\1", sanitized)
        sanitized = _BARE_KEY_PATTERN.sub(r'\g<prefix>"\g<key>"\g<suffix>', sanitized)
        logger.debug(
            "Sanitized structured JSON candidate: before=%s after=%s",
            IncrementalJsonlParser._preview_text(candidate),
            IncrementalJsonlParser._preview_text(sanitized),
        )
        return sanitized

    @staticmethod
    def _decode_json_like_text(candidate: str) -> dict | list:
        json_error_message = ""
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as json_exc:
            json_error_message = json_exc.msg
            logger.debug(
                "Standard JSON decode failed, trying literal_eval fallback: preview=%s",
                IncrementalJsonlParser._preview_text(candidate),
            )
        else:
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Decoded content must be a JSON object or array.")
            return parsed

        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError) as literal_exc:
            raise ValueError(json_error_message or str(literal_exc)) from literal_exc

        if not isinstance(parsed, (dict, list)):
            raise ValueError("Decoded content must be a JSON object or array.")

        logger.info(
            "Decoded structured JSON candidate with literal_eval fallback: parsed_type=%s",
            type(parsed).__name__,
        )
        return parsed

    @staticmethod
    def _expand_payload(
        parsed: dict | list,
        *,
        raw_line: str,
    ) -> list[JsonlParseEvent]:
        if isinstance(parsed, dict):
            logger.debug("Expanded structured JSON payload into 1 event")
            return [JsonlParseEvent(parsed=parsed, raw_line=raw_line)]

        events: list[JsonlParseEvent] = []
        for item in parsed:
            if not isinstance(item, (dict, list)):
                raise JsonlStreamParseError(
                    "When a JSON payload is a JSON array, each element must be a JSON object or array."
                )
            events.append(JsonlParseEvent(parsed=item, raw_line=raw_line))
        logger.info(
            "Expanded structured JSON array payload: event_count=%s preview=%s",
            len(events),
            IncrementalJsonlParser._preview_text(raw_line),
        )
        return events

    @staticmethod
    def _preview_text(value: str, max_length: int = 120) -> str:
        single_line = " ".join(value.split())
        if len(single_line) <= max_length:
            return single_line
        return single_line[: max_length - 3] + "..."

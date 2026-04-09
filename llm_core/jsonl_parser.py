from __future__ import annotations

import json

from llm_core.exceptions import JsonlStreamParseError
from llm_core.types import JsonlParseEvent


class IncrementalJsonlParser:
    """Incrementally parse JSONL content from arbitrarily split chunks."""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_code_fence = False

    @property
    def buffer(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> list[JsonlParseEvent]:
        self._buffer += chunk
        return self._drain_lines(include_tail=False)

    def flush(self) -> list[JsonlParseEvent]:
        return self._drain_lines(include_tail=True)

    def _drain_lines(self, *, include_tail: bool) -> list[JsonlParseEvent]:
        events: list[JsonlParseEvent] = []

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index < 0:
                break

            raw_line = self._buffer[:newline_index]
            self._buffer = self._buffer[newline_index + 1 :]
            event = self._parse_line(raw_line)
            if event is not None:
                events.append(event)

        if include_tail and self._buffer.strip():
            raw_line = self._buffer
            self._buffer = ""
            event = self._parse_line(raw_line)
            if event is not None:
                events.append(event)

        return events

    def _parse_line(self, raw_line: str) -> JsonlParseEvent | None:
        candidate = raw_line.strip()
        if not candidate:
            return None
        if self._is_code_fence_line(candidate):
            return None

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise JsonlStreamParseError(
                f"Failed to decode JSONL line: {exc.msg} (line={candidate!r})"
            ) from exc

        if not isinstance(parsed, (dict, list)):
            raise JsonlStreamParseError(
                "Each JSONL line must decode to a JSON object or array."
            )

        return JsonlParseEvent(parsed=parsed, raw_line=candidate)

    @staticmethod
    def _is_code_fence_line(candidate: str) -> bool:
        normalized = candidate.lower()
        return normalized == "```" or normalized.startswith("```json")

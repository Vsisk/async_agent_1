from __future__ import annotations

import json

from llm_core.exceptions import JsonlStreamParseError
from llm_core.types import JsonlParseEvent


class IncrementalJsonlParser:
    """Incrementally parse JSONL content from arbitrarily split chunks."""

    def __init__(self) -> None:
        self._buffer = ""

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
            events.extend(self._parse_line(raw_line))

        if include_tail and self._buffer.strip():
            raw_line = self._buffer
            self._buffer = ""
            events.extend(self._parse_line(raw_line))

        return events

    def _parse_line(self, raw_line: str) -> list[JsonlParseEvent]:
        candidate = raw_line.strip()
        if not candidate:
            return []
        if self._is_code_fence_line(candidate):
            return []
        candidate = self._sanitize_candidate(candidate)
        if not candidate:
            return []

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

        return self._expand_jsonl_payload(parsed, raw_line=candidate)

    @staticmethod
    def _is_code_fence_line(candidate: str) -> bool:
        normalized = candidate.lower()
        return normalized == "```" or normalized == "```json"

    @staticmethod
    def _sanitize_candidate(candidate: str) -> str:
        if not candidate.startswith("```") or not candidate.endswith("```"):
            return candidate

        inner_content = candidate[3:-3].strip()
        if not inner_content:
            return ""

        lowered = inner_content.lower()
        if lowered.startswith("json"):
            inner_content = inner_content[4:].strip()

        return inner_content.strip()

    @staticmethod
    def _expand_jsonl_payload(
        parsed: dict | list,
        *,
        raw_line: str,
    ) -> list[JsonlParseEvent]:
        if isinstance(parsed, dict):
            return [JsonlParseEvent(parsed=parsed, raw_line=raw_line)]

        events: list[JsonlParseEvent] = []
        for item in parsed:
            if not isinstance(item, (dict, list)):
                raise JsonlStreamParseError(
                    "When a JSONL line is a JSON array, each element must be a JSON object or array."
                )
            events.append(JsonlParseEvent(parsed=item, raw_line=raw_line))
        return events

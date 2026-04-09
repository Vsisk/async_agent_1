from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from llm_core.exceptions import (
    PromptLanguageNotFoundError,
    PromptNotFoundError,
    PromptRenderError,
)
from llm_core.types import PromptCatalog, PromptDefinition, PromptRenderResult

_VARIABLE_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
logger = logging.getLogger(__name__)


def extract_template_variables(template_text: str) -> set[str]:
    """Return all variable names referenced by {{variable}} placeholders."""
    return set(_VARIABLE_PATTERN.findall(template_text))


def render_template_text(
    template_text: str,
    variables: Mapping[str, Any],
    *,
    strict: bool = True,
) -> str:
    """Pure-function template renderer used by PromptManager and unit tests."""
    logger.debug(
        "Rendering template text: template_length=%s variable_count=%s strict=%s",
        len(template_text),
        len(variables),
        strict,
    )

    required_variables = extract_template_variables(template_text)
    missing_variables = sorted(
        name for name in required_variables if name not in variables or variables[name] is None
    )
    if strict and missing_variables:
        raise PromptRenderError(
            f"Missing prompt variables: {', '.join(missing_variables)}"
        )

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = variables.get(key, "")
        return "" if value is None else str(value)

    return _VARIABLE_PATTERN.sub(replace, template_text)


class PromptManager:
    """Load prompt.json and render prompt templates by name."""

    def __init__(
        self,
        prompt_file: str | Path | None = None,
        *,
        default_lang: str = "zh",
        fallback_lang: str = "en",
    ) -> None:
        self._prompt_file = Path(prompt_file or Path(__file__).with_name("prompt.json"))
        self._default_lang = default_lang
        self._fallback_lang = fallback_lang
        self._catalog = self._load_catalog()

    @property
    def prompt_file(self) -> Path:
        return self._prompt_file

    def get_prompt(self, prompt_name: str, *, lang: str | None = None) -> str:
        logger.debug("Loading prompt template: prompt_name=%s lang=%s", prompt_name, lang)
        prompt_definition = self._catalog.prompts.get(prompt_name)
        if prompt_definition is None:
            raise PromptNotFoundError(f"Prompt template not found: {prompt_name}")

        selected_lang = lang or self._default_lang
        prompt_text = getattr(prompt_definition, selected_lang, None)
        if prompt_text:
            return prompt_text

        fallback_text = getattr(prompt_definition, self._fallback_lang, None)
        if fallback_text:
            return fallback_text

        raise PromptLanguageNotFoundError(
            f"Prompt '{prompt_name}' does not provide language '{selected_lang}' "
            f"or fallback '{self._fallback_lang}'."
        )

    def build_prompt_text(self, prompt_template: list[str], *, lang: str | None = None) -> str:
        logger.debug(
            "Building combined prompt text: templates=%s lang=%s",
            prompt_template,
            lang,
        )
        prompt_parts = [self.get_prompt(prompt_name, lang=lang) for prompt_name in prompt_template]
        return "\n\n".join(prompt_parts)

    def render(
        self,
        prompt_template: list[str],
        *,
        lang: str | None = None,
        variables: Mapping[str, Any] | None = None,
        strict: bool = True,
    ) -> PromptRenderResult:
        if not prompt_template:
            raise PromptRenderError("prompt_template must contain at least one template name.")

        logger.info(
            "Rendering prompt: templates=%s lang=%s strict=%s variable_keys=%s",
            prompt_template,
            lang or self._default_lang,
            strict,
            sorted((variables or {}).keys()),
        )
        render_variables = dict(variables or {})
        combined_prompt = self.build_prompt_text(prompt_template, lang=lang)
        prompt_text = render_template_text(combined_prompt, render_variables, strict=strict)
        logger.debug(
            "Rendered prompt completed: templates=%s prompt_length=%s",
            prompt_template,
            len(prompt_text),
        )

        return PromptRenderResult(
            prompt_text=prompt_text,
            used_templates=list(prompt_template),
            variables=render_variables,
        )

    def _load_catalog(self) -> PromptCatalog:
        logger.info("Loading prompt catalog from %s", self._prompt_file)
        try:
            payload = json.loads(self._prompt_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PromptRenderError(f"Prompt file not found: {self._prompt_file}") from exc
        except json.JSONDecodeError as exc:
            raise PromptRenderError(
                f"Prompt file is not valid JSON: {self._prompt_file}"
            ) from exc

        if not isinstance(payload, dict):
            raise PromptRenderError("prompt.json must be a JSON object keyed by prompt name.")

        prompts: dict[str, PromptDefinition] = {}
        for prompt_name, value in payload.items():
            if not isinstance(value, dict):
                raise PromptRenderError(
                    f"Prompt '{prompt_name}' must map to an object containing zh/en templates."
                )
            prompts[prompt_name] = PromptDefinition.from_mapping(value)

        logger.info("Loaded prompt catalog: prompt_count=%s", len(prompts))
        return PromptCatalog(prompts=prompts)

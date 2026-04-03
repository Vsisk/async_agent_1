from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_core.jsonl_parser import IncrementalJsonlParser
from llm_core.llm_client import OpenAILLMClient

ItemProcessStatus = Literal["success", "failed", "cancelled"]


@dataclass(slots=True)
class SectionContext:
    section_id: str
    section_image: Any | None = None
    image_path: str | None = None
    image_bytes: bytes | None = None
    section_bbox: tuple[float, float, float, float] | None = None
    page_no: int | None = None
    upstream_metadata: dict[str, Any] = field(default_factory=dict)
    classification_result: dict[str, Any] = field(default_factory=dict)
    section_type: str | None = None


@dataclass(slots=True)
class SectionItem:
    item_id: str
    item_type: str
    raw_content: str
    bbox: tuple[float, float, float, float] | None = None
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ItemProcessResult:
    item: SectionItem
    parsed_content: dict[str, Any] | list[Any] | str | None
    description: str | None
    status: ItemProcessStatus
    error: str | None = None


@dataclass(slots=True)
class StructuredNode:
    section_id: str
    section_type: str
    node_type: str
    children: list[dict[str, Any]] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    item_results: list[ItemProcessResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


ClassifierFn = Callable[[SectionContext], str | Awaitable[str]]
ItemStreamFn = Callable[[SectionContext, str], AsyncIterator[str]]
ParseItemContentFn = Callable[
    [SectionItem, SectionContext],
    dict[str, Any] | list[Any] | str | Awaitable[dict[str, Any] | list[Any] | str],
]
GenerateItemDescriptionFn = Callable[
    [SectionItem, dict[str, Any] | list[Any] | str, SectionContext],
    str | Awaitable[str],
]


class SchedulerHaltedError(RuntimeError):
    """Raised when scheduler stops accepting new item tasks."""


class SectionPipelineExecutionError(RuntimeError):
    """Raised when fail_fast is enabled and item failures occur."""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(slots=True)
class JsonlIncrementalItemParser:
    """Parse JSONL chunks into SectionItem objects."""

    item_id_prefix: str = "item"
    _delegate: IncrementalJsonlParser = field(init=False, repr=False)
    _auto_order: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._delegate = IncrementalJsonlParser()
        self._auto_order = 0

    def feed(self, chunk: str) -> list[SectionItem]:
        events = self._delegate.feed(chunk)
        return [self._to_item(event.parsed) for event in events]

    def flush(self) -> list[SectionItem]:
        events = self._delegate.flush()
        return [self._to_item(event.parsed) for event in events]

    def _to_item(self, payload: dict[str, Any] | list[Any]) -> SectionItem:
        if not isinstance(payload, dict):
            raise ValueError("Each JSONL line must decode to object for SectionItem mapping.")

        raw_order = payload.get("order")
        order = raw_order if isinstance(raw_order, int) else self._auto_order

        raw_item_id = payload.get("item_id")
        item_id = str(raw_item_id) if raw_item_id else f"{self.item_id_prefix}-{order}"

        item_type = str(payload.get("item_type") or "unknown")
        raw_content = str(payload.get("raw_content") or "")

        bbox_raw = payload.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = tuple(float(x) for x in bbox_raw)

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"item_id", "item_type", "raw_content", "bbox", "order"}
        }

        self._auto_order = max(self._auto_order + 1, order + 1)
        return SectionItem(
            item_id=item_id,
            item_type=item_type,
            raw_content=raw_content,
            bbox=bbox,
            order=order,
            metadata=metadata,
        )


@dataclass(slots=True)
class DefaultItemProcessor:
    parse_item_content: ParseItemContentFn
    generate_item_description: GenerateItemDescriptionFn

    async def process_item(
        self,
        item: SectionItem,
        context: SectionContext,
    ) -> ItemProcessResult:
        try:
            parsed_content = await _maybe_await(self.parse_item_content(item, context))
            description = await _maybe_await(
                self.generate_item_description(item, parsed_content, context)
            )
            return ItemProcessResult(
                item=item,
                parsed_content=parsed_content,
                description=description,
                status="success",
            )
        except Exception as exc:
            return ItemProcessResult(
                item=item,
                parsed_content=None,
                description=None,
                status="failed",
                error=str(exc),
            )


@dataclass(slots=True)
class AsyncioItemTaskScheduler:
    max_concurrency: int = 5
    fail_fast: bool = False
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _tasks: set[asyncio.Task[None]] = field(init=False, repr=False)
    _results: list[ItemProcessResult] = field(init=False, repr=False)
    _halted: bool = field(init=False, repr=False)
    _had_failure: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._results: list[ItemProcessResult] = []
        self._halted = False
        self._had_failure = False

    @property
    def has_failures(self) -> bool:
        return self._had_failure

    def add_item_task(
        self,
        item: SectionItem,
        context: SectionContext,
        item_processor: DefaultItemProcessor,
    ) -> None:
        if self._halted:
            raise SchedulerHaltedError("Scheduler is halted and no longer accepts tasks.")

        task = asyncio.create_task(self._run_one(item, context, item_processor))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_all(self) -> list[ItemProcessResult]:
        while self._tasks:
            current_tasks = list(self._tasks)
            await asyncio.gather(*current_tasks, return_exceptions=True)
        return list(self._results)

    async def _run_one(
        self,
        item: SectionItem,
        context: SectionContext,
        item_processor: DefaultItemProcessor,
    ) -> None:
        try:
            async with self._semaphore:
                result = await item_processor.process_item(item, context)
        except asyncio.CancelledError:
            self._results.append(
                ItemProcessResult(
                    item=item,
                    parsed_content=None,
                    description=None,
                    status="cancelled",
                    error="Task cancelled by scheduler.",
                )
            )
            raise
        except Exception as exc:
            self._had_failure = True
            self._results.append(
                ItemProcessResult(
                    item=item,
                    parsed_content=None,
                    description=None,
                    status="failed",
                    error=str(exc),
                )
            )
            if self.fail_fast:
                self._halted = True
                self._cancel_all_pending()
            return

        if result.status != "success":
            self._had_failure = True
            if self.fail_fast:
                self._halted = True
                self._cancel_all_pending()

        self._results.append(result)

    def _cancel_all_pending(self) -> None:
        for task in list(self._tasks):
            if not task.done():
                task.cancel()


@dataclass(slots=True)
class DefaultStructuredNodeAggregator:
    node_type: str = "structured_section"

    async def aggregate(
        self,
        context: SectionContext,
        section_type: str,
        item_results: list[ItemProcessResult],
    ) -> StructuredNode:
        ordered_results = sorted(item_results, key=lambda result: result.item.order)

        children = [
            {
                "item_id": result.item.item_id,
                "item_type": result.item.item_type,
                "order": result.item.order,
                "status": result.status,
                "description": result.description,
                "parsed_content": result.parsed_content,
                "error": result.error,
            }
            for result in ordered_results
        ]

        success_count = sum(1 for result in ordered_results if result.status == "success")
        failed_count = sum(1 for result in ordered_results if result.status == "failed")
        cancelled_count = sum(
            1 for result in ordered_results if result.status == "cancelled"
        )

        return StructuredNode(
            section_id=context.section_id,
            section_type=section_type,
            node_type=self.node_type,
            children=children,
            fields={
                "total_items": len(ordered_results),
                "success_count": success_count,
                "failed_count": failed_count,
                "cancelled_count": cancelled_count,
            },
            item_results=ordered_results,
            metadata={
                "page_no": context.page_no,
                "section_bbox": context.section_bbox,
                "classification_result": context.classification_result,
                "upstream_metadata": context.upstream_metadata,
            },
        )


@dataclass(slots=True)
class OpenAISectionItemStreamer:
    llm_client: OpenAILLMClient
    prompt_template: list[str]
    model: str = "gpt-4.1"
    temperature: float = 0
    lang: str = "zh"

    async def stream_items(
        self,
        context: SectionContext,
        section_type: str,
    ) -> AsyncIterator[str]:
        section_content = str(context.upstream_metadata.get("section_content", ""))
        stream = self.llm_client.generate_result_by_llm(
            prompt_template=self.prompt_template,
            stream=True,
            model=self.model,
            temperature=self.temperature,
            lang=self.lang,
            section_type=section_type,
            section_content=section_content,
        )
        async for event in stream:
            yield json.dumps(event.object, ensure_ascii=False) + "\n"


async def parse_section(
    section_context: SectionContext,
    *,
    classifier: ClassifierFn,
    item_streamer: ItemStreamFn,
    parse_item_content: ParseItemContentFn,
    generate_item_description: GenerateItemDescriptionFn,
    max_concurrency: int = 5,
    fail_fast: bool = False,
    item_parser: JsonlIncrementalItemParser | None = None,
    aggregator: DefaultStructuredNodeAggregator | None = None,
) -> StructuredNode:
    parser = item_parser or JsonlIncrementalItemParser()
    scheduler = AsyncioItemTaskScheduler(
        max_concurrency=max_concurrency,
        fail_fast=fail_fast,
    )
    processor = DefaultItemProcessor(
        parse_item_content=parse_item_content,
        generate_item_description=generate_item_description,
    )
    final_aggregator = aggregator or DefaultStructuredNodeAggregator()

    section_type_raw = await _maybe_await(classifier(section_context))
    section_type = str(section_type_raw)
    section_context.section_type = section_type
    section_context.classification_result["section_type"] = section_type

    halted = False
    async for chunk in item_streamer(section_context, section_type):
        for item in parser.feed(chunk):
            try:
                scheduler.add_item_task(item, section_context, processor)
            except SchedulerHaltedError:
                halted = True
                break
        if halted:
            break

    if not halted:
        for item in parser.flush():
            try:
                scheduler.add_item_task(item, section_context, processor)
            except SchedulerHaltedError:
                break

    item_results = await scheduler.wait_all()
    if fail_fast and scheduler.has_failures:
        raise SectionPipelineExecutionError(
            "Section pipeline failed in fail_fast mode due to item processing failure."
        )

    return await final_aggregator.aggregate(section_context, section_type, item_results)

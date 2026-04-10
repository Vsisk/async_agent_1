from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from llm_core.jsonl_parser import IncrementalJsonlParser
from llm_core.llm_client import OpenAILLMClient

logger = logging.getLogger(__name__)

ItemProcessStatus = Literal["success", "failed", "cancelled"]
SectionType = Literal["key_value", "plain_text", "table"]
ItemKind = Literal["kv", "text", "col", "row"]
LogicDataType = Literal["field", "summaryField", "commonField", "rowRequirements"]

SECTION_ITEM_KINDS: dict[SectionType, tuple[ItemKind, ...]] = {
    "key_value": ("kv",),
    "plain_text": ("kv", "text"),
    "table": ("col", "row"),
}


@dataclass(slots=True)
class SectionShell:
    node_id: str = ""
    type: SectionType | None = None
    name: str = ""
    annotation: str = ""
    reference_list: list["SectionReference"] = field(default_factory=list)


@dataclass(slots=True)
class SectionRuntime:
    section: SectionShell
    image_base64: str
    upstream_metadata: dict[str, Any] = field(default_factory=dict)
    classification_result: dict[str, Any] = field(default_factory=dict)
    table_columns: list["SectionItem"] = field(default_factory=list)
    processed_table_columns: list["ItemProcessResult"] = field(default_factory=list)
    table_column_context: list[dict[str, str]] = field(default_factory=list)
    section_bbox: tuple[float, float, float, float] | None = None
    page_no: int | None = None


@dataclass(slots=True)
class KeyValuePayload:
    key: str
    exp: str
    annotation: str
    cbs_name: str


@dataclass(slots=True)
class TextPayload:
    text: str


@dataclass(slots=True)
class RowPayload:
    key: str
    annotation: str


ItemPayload = KeyValuePayload | TextPayload | RowPayload


@dataclass(slots=True)
class SectionItem:
    item_id: str
    item_kind: ItemKind
    payload: ItemPayload
    bbox: tuple[float, float, float, float] | None = None
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogicDataNode:
    node_id: str
    type: LogicDataType
    name: str
    annotation: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SectionReference:
    pdf_name: str
    reference_node_id: str
    pdf_exp: str
    cbs_name: str


@dataclass(slots=True)
class ItemProcessingOutput:
    logic_data_nodes: list[LogicDataNode] = field(default_factory=list)
    section_annotation_text: str | None = None
    reference_list: list[SectionReference] = field(default_factory=list)
    extra_json: dict[str, Any] | None = None


@dataclass(slots=True)
class ItemProcessResult:
    item: SectionItem
    output: ItemProcessingOutput | None
    status: ItemProcessStatus
    error: str | None = None


@dataclass(slots=True)
class ProcessedItem:
    item_id: str
    item_kind: ItemKind
    order: int
    payload: dict[str, Any]
    status: ItemProcessStatus
    output: dict[str, Any] | None
    error: str | None = None


@dataclass(slots=True)
class SectionDraft:
    node_id: str
    type: SectionType
    name: str
    annotation: str
    reference_list: list[SectionReference] = field(default_factory=list)


@dataclass(slots=True)
class ParsedSectionResult:
    section: SectionDraft
    logic_data_nodes: list[LogicDataNode] = field(default_factory=list)
    item_results: list[ItemProcessResult] = field(default_factory=list)
    items: list[ProcessedItem] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


ClassifierFn = Callable[[SectionShell, str], SectionType | Awaitable[SectionType]]
ItemStreamFn = Callable[[SectionShell, str, SectionType], AsyncIterator[str]]
KVItemHandlerFn = Callable[[SectionItem, SectionRuntime], ItemProcessingOutput | Awaitable[ItemProcessingOutput]]
TextItemHandlerFn = Callable[[SectionItem, SectionRuntime], ItemProcessingOutput | Awaitable[ItemProcessingOutput]]
ColItemHandlerFn = Callable[[SectionItem, SectionRuntime], ItemProcessingOutput | Awaitable[ItemProcessingOutput]]
RowItemHandlerFn = Callable[[SectionItem, SectionRuntime], ItemProcessingOutput | Awaitable[ItemProcessingOutput]]


class SchedulerHaltedError(RuntimeError):
    """Raised when scheduler stops accepting new item tasks."""


class SectionPipelineExecutionError(RuntimeError):
    """Raised when fail_fast is enabled and item failures occur."""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_section_type(value: Any) -> SectionType:
    normalized = str(value)
    if normalized not in SECTION_ITEM_KINDS:
        raise ValueError(
            "Unsupported section_type. Expected one of: key_value, plain_text, table."
        )
    return normalized  # type: ignore[return-value]


def _read_required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return "" if value is None else str(value)


def _item_payload_to_dict(payload: ItemPayload) -> dict[str, Any]:
    return asdict(payload)


def _build_logic_node_id(section_id: str, item_id: str, index: int = 0) -> str:
    return f"{section_id}:{item_id}" if index == 0 else f"{section_id}:{item_id}:{index}"


def _ensure_section_shell_defaults(
    section: SectionShell,
    upstream_metadata: dict[str, Any],
) -> None:
    if not section.node_id:
        section.node_id = str(upstream_metadata.get("section_node_id") or "section")
        logger.debug("Applied default section node_id: node_id=%s", section.node_id)


def _get_pdf_name(runtime: SectionRuntime) -> str:
    value = runtime.upstream_metadata.get("pdf_name") or runtime.upstream_metadata.get(
        "source_pdf_name"
    )
    return "" if value is None else str(value)


@dataclass(slots=True)
class JsonlIncrementalItemParser:
    """Parse JSONL chunks into SectionItem objects."""

    item_id_prefix: str = "item"
    _delegate: IncrementalJsonlParser = field(init=False, repr=False)
    _auto_order: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._delegate = IncrementalJsonlParser()
        self._auto_order = 0
        logger.debug("Initialized JsonlIncrementalItemParser: item_id_prefix=%s", self.item_id_prefix)

    def feed(self, chunk: str) -> list[SectionItem]:
        logger.debug("Parsing section items from chunk: chunk_length=%s", len(chunk))
        events = self._delegate.feed(chunk)
        items = [self._to_item(event.parsed) for event in events]
        logger.debug("Parsed section items from chunk: item_count=%s", len(items))
        return items

    def flush(self) -> list[SectionItem]:
        logger.debug("Flushing section item parser")
        events = self._delegate.flush()
        items = [self._to_item(event.parsed) for event in events]
        logger.debug("Flushed section item parser: item_count=%s", len(items))
        return items

    def _to_item(self, payload: dict[str, Any] | list[Any]) -> SectionItem:
        if not isinstance(payload, dict):
            raise ValueError("Each JSONL line must decode to object for SectionItem mapping.")

        raw_order = payload.get("order")
        order = raw_order if isinstance(raw_order, int) else self._auto_order

        raw_item_id = payload.get("item_id")
        item_id = str(raw_item_id) if raw_item_id else f"{self.item_id_prefix}-{order}"

        raw_kind = str(payload.get("item_kind") or "")
        if raw_kind not in {"kv", "text", "col", "row"}:
            raise ValueError("item_kind must be one of: kv, text, col, row.")
        item_kind: ItemKind = raw_kind  # type: ignore[assignment]

        bbox_raw = payload.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = tuple(float(x) for x in bbox_raw)

        item_payload = self._build_payload(item_kind, payload)
        metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "item_id",
                "item_kind",
                "key",
                "exp",
                "cbs_name",
                "text",
                "annotation",
                "bbox",
                "order",
            }
        }

        self._auto_order = max(self._auto_order + 1, order + 1)
        item = SectionItem(
            item_id=item_id,
            item_kind=item_kind,
            payload=item_payload,
            bbox=bbox,
            order=order,
            metadata=metadata,
        )
        logger.info(
            "Built SectionItem: item_id=%s item_kind=%s order=%s metadata_keys=%s",
            item.item_id,
            item.item_kind,
            item.order,
            sorted(item.metadata.keys()),
        )
        return item

    def _build_payload(self, item_kind: ItemKind, payload: dict[str, Any]) -> ItemPayload:
        if item_kind in {"kv", "col"}:
            annotation = _read_required_string(payload, "annotation")
            if not annotation:
                annotation = _read_required_string(payload, "exp")
            return KeyValuePayload(
                key=_read_required_string(payload, "key"),
                exp=_read_required_string(payload, "exp"),
                annotation=annotation,
                cbs_name=_read_required_string(payload, "cbs_name"),
            )

        if item_kind == "text":
            return TextPayload(text=_read_required_string(payload, "text"))

        return RowPayload(
            key=_read_required_string(payload, "key"),
            annotation=_read_required_string(payload, "annotation"),
        )


@dataclass(slots=True)
class DefaultItemProcessor:
    process_kv_item: KVItemHandlerFn
    process_text_item: TextItemHandlerFn
    process_col_item: ColItemHandlerFn
    process_row_item: RowItemHandlerFn

    async def process_item(
        self,
        item: SectionItem,
        runtime: SectionRuntime,
    ) -> ItemProcessResult:
        logger.info(
            "Processing item: section_id=%s item_id=%s item_kind=%s",
            runtime.section.node_id,
            item.item_id,
            item.item_kind,
        )
        try:
            if item.item_kind == "kv":
                output = await _maybe_await(self.process_kv_item(item, runtime))
            elif item.item_kind == "text":
                output = await _maybe_await(self.process_text_item(item, runtime))
            elif item.item_kind == "col":
                output = await _maybe_await(self.process_col_item(item, runtime))
            else:
                output = await _maybe_await(self.process_row_item(item, runtime))

            logger.info(
                "Processed item successfully: section_id=%s item_id=%s item_kind=%s logic_nodes=%s references=%s",
                runtime.section.node_id,
                item.item_id,
                item.item_kind,
                len(output.logic_data_nodes),
                len(output.reference_list),
            )
            return ItemProcessResult(item=item, output=output, status="success")
        except Exception as exc:
            logger.exception(
                "Processing item failed: section_id=%s item_id=%s item_kind=%s",
                runtime.section.node_id,
                item.item_id,
                item.item_kind,
            )
            return ItemProcessResult(
                item=item,
                output=None,
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
        self._tasks = set()
        self._results = []
        self._halted = False
        self._had_failure = False
        logger.info(
            "Initialized AsyncioItemTaskScheduler: max_concurrency=%s fail_fast=%s",
            self.max_concurrency,
            self.fail_fast,
        )

    @property
    def has_failures(self) -> bool:
        return self._had_failure

    def add_item_task(
        self,
        item: SectionItem,
        runtime: SectionRuntime,
        item_processor: DefaultItemProcessor,
    ) -> None:
        if self._halted:
            raise SchedulerHaltedError("Scheduler is halted and no longer accepts tasks.")

        logger.debug(
            "Scheduling item task: section_id=%s item_id=%s item_kind=%s outstanding_tasks=%s",
            runtime.section.node_id,
            item.item_id,
            item.item_kind,
            len(self._tasks),
        )
        task = asyncio.create_task(self._run_one(item, runtime, item_processor))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_all(self) -> list[ItemProcessResult]:
        logger.info("Waiting for all scheduled item tasks: task_count=%s", len(self._tasks))
        while self._tasks:
            current_tasks = list(self._tasks)
            await asyncio.gather(*current_tasks, return_exceptions=True)
        logger.info("All scheduled item tasks completed: result_count=%s", len(self._results))
        return list(self._results)

    async def _run_one(
        self,
        item: SectionItem,
        runtime: SectionRuntime,
        item_processor: DefaultItemProcessor,
    ) -> None:
        try:
            async with self._semaphore:
                logger.debug(
                    "Running item task: section_id=%s item_id=%s item_kind=%s",
                    runtime.section.node_id,
                    item.item_id,
                    item.item_kind,
                )
                result = await item_processor.process_item(item, runtime)
        except asyncio.CancelledError:
            logger.warning(
                "Item task cancelled: section_id=%s item_id=%s item_kind=%s",
                runtime.section.node_id,
                item.item_id,
                item.item_kind,
            )
            self._results.append(
                ItemProcessResult(
                    item=item,
                    output=None,
                    status="cancelled",
                    error="Task cancelled by scheduler.",
                )
            )
            raise
        except Exception as exc:
            self._had_failure = True
            logger.exception(
                "Item task crashed: section_id=%s item_id=%s item_kind=%s",
                runtime.section.node_id,
                item.item_id,
                item.item_kind,
            )
            self._results.append(
                ItemProcessResult(
                    item=item,
                    output=None,
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
            logger.warning(
                "Item task finished with non-success status: section_id=%s item_id=%s status=%s",
                runtime.section.node_id,
                item.item_id,
                result.status,
            )
            if self.fail_fast:
                self._halted = True
                self._cancel_all_pending()

        self._results.append(result)
        logger.debug(
            "Recorded item task result: section_id=%s item_id=%s status=%s total_results=%s",
            runtime.section.node_id,
            item.item_id,
            result.status,
            len(self._results),
        )

    def _cancel_all_pending(self) -> None:
        logger.warning("Cancelling all pending item tasks: task_count=%s", len(self._tasks))
        for task in list(self._tasks):
            if not task.done():
                task.cancel()


def _validate_item_for_section(section_type: SectionType, item: SectionItem) -> bool:
    allowed_kinds = SECTION_ITEM_KINDS[section_type]
    if item.item_kind not in allowed_kinds:
        logger.warning(
            "Skipping item with incompatible kind: section_type=%s item_id=%s item_kind=%s allowed_kinds=%s",
            section_type,
            item.item_id,
            item.item_kind,
            allowed_kinds,
        )
        return False
    return True


def _result_to_processed_item(result: ItemProcessResult) -> ProcessedItem:
    return ProcessedItem(
        item_id=result.item.item_id,
        item_kind=result.item.item_kind,
        order=result.item.order,
        payload=_item_payload_to_dict(result.item.payload),
        status=result.status,
        output=asdict(result.output) if result.output is not None else None,
        error=result.error,
    )


def _sync_table_columns(
    runtime: SectionRuntime,
    item_results: list[ItemProcessResult],
) -> None:
    col_results = [
        result
        for result in item_results
        if result.item.item_kind == "col" and result.status == "success"
    ]
    runtime.processed_table_columns = list(col_results)
    runtime.table_columns = [result.item for result in col_results]
    runtime.table_column_context = [
        {
            "item_id": result.item.item_id,
            "cbs_name": result.item.payload.cbs_name,
            "annotation": result.item.payload.annotation,
        }
        for result in col_results
        if isinstance(result.item.payload, KeyValuePayload)
    ]
    logger.info(
        "Synchronized table columns: section_id=%s column_count=%s",
        runtime.section.node_id,
        len(runtime.table_columns),
    )


def _default_kv_handler(item: SectionItem, runtime: SectionRuntime) -> ItemProcessingOutput:
    payload = item.payload
    if not isinstance(payload, KeyValuePayload):
        raise TypeError("kv item expects KeyValuePayload.")

    node = LogicDataNode(
        node_id=_build_logic_node_id(runtime.section.node_id, item.item_id),
        type="field",
        name=payload.cbs_name,
        annotation=payload.annotation,
    )
    reference = SectionReference(
        pdf_name=_get_pdf_name(runtime),
        reference_node_id=node.node_id,
        pdf_exp=payload.annotation,
        cbs_name=payload.cbs_name,
    )
    return ItemProcessingOutput(logic_data_nodes=[node], reference_list=[reference])


def _default_text_handler(item: SectionItem, runtime: SectionRuntime) -> ItemProcessingOutput:
    del runtime
    payload = item.payload
    if not isinstance(payload, TextPayload):
        raise TypeError("text item expects TextPayload.")

    return ItemProcessingOutput(section_annotation_text=f"固定文本: {payload.text}")


def _default_col_handler(item: SectionItem, runtime: SectionRuntime) -> ItemProcessingOutput:
    payload = item.payload
    if not isinstance(payload, KeyValuePayload):
        raise TypeError("col item expects KeyValuePayload.")

    is_summary = bool(item.metadata.get("is_summary"))
    node_type: LogicDataType = "summaryField" if is_summary else "commonField"
    node = LogicDataNode(
        node_id=_build_logic_node_id(runtime.section.node_id, item.item_id),
        type=node_type,
        name=payload.cbs_name,
        annotation=payload.annotation,
        data={"pdf_key": payload.key},
    )
    reference = SectionReference(
        pdf_name=_get_pdf_name(runtime),
        reference_node_id=node.node_id,
        pdf_exp=payload.annotation,
        cbs_name=payload.cbs_name,
    )
    return ItemProcessingOutput(logic_data_nodes=[node], reference_list=[reference])


def _default_row_handler(item: SectionItem, runtime: SectionRuntime) -> ItemProcessingOutput:
    payload = item.payload
    if not isinstance(payload, RowPayload):
        raise TypeError("row item expects RowPayload.")

    column_requirements = item.metadata.get("column_requirements")
    if column_requirements is None:
        column_requirements = {}

    node = LogicDataNode(
        node_id=_build_logic_node_id(runtime.section.node_id, item.item_id),
        type="rowRequirements",
        name=payload.key,
        annotation=payload.annotation,
        data={
            "column_requirements": column_requirements,
        },
    )
    reference = SectionReference(
        pdf_name=_get_pdf_name(runtime),
        reference_node_id=node.node_id,
        pdf_exp=payload.annotation,
        cbs_name=payload.key,
    )
    return ItemProcessingOutput(
        logic_data_nodes=[node],
        reference_list=[reference],
        extra_json={"column_requirements": column_requirements},
    )


@dataclass(slots=True)
class ParsedSectionAggregator:
    async def aggregate(
        self,
        runtime: SectionRuntime,
        section_type: SectionType,
        item_results: list[ItemProcessResult],
    ) -> ParsedSectionResult:
        logger.info(
            "Aggregating parsed section: section_id=%s section_type=%s item_result_count=%s",
            runtime.section.node_id,
            section_type,
            len(item_results),
        )
        ordered_results = sorted(item_results, key=lambda result: result.item.order)
        success_count = sum(1 for result in ordered_results if result.status == "success")
        failed_count = sum(1 for result in ordered_results if result.status == "failed")
        cancelled_count = sum(
            1 for result in ordered_results if result.status == "cancelled"
        )

        logic_data_nodes: list[LogicDataNode] = []
        reference_list: list[SectionReference] = []
        annotation_parts: list[str] = []

        for result in ordered_results:
            if result.status != "success" or result.output is None:
                continue
            logic_data_nodes.extend(result.output.logic_data_nodes)
            reference_list.extend(result.output.reference_list)
            if result.output.section_annotation_text:
                annotation_parts.append(result.output.section_annotation_text)

        section = SectionDraft(
            node_id=runtime.section.node_id,
            type=section_type,
            name=runtime.section.name,
            annotation=" ".join(part for part in annotation_parts if part).strip(),
            reference_list=reference_list,
        )

        aggregated = ParsedSectionResult(
            section=section,
            logic_data_nodes=logic_data_nodes,
            item_results=ordered_results,
            items=[_result_to_processed_item(result) for result in ordered_results],
            stats={
                "total_items": len(ordered_results),
                "success_count": success_count,
                "failed_count": failed_count,
                "cancelled_count": cancelled_count,
            },
            metadata={
                "image_base64": runtime.image_base64,
                "page_no": runtime.page_no,
                "section_bbox": runtime.section_bbox,
                "classification_result": runtime.classification_result,
                "upstream_metadata": runtime.upstream_metadata,
            },
        )
        logger.info(
            "Aggregated parsed section completed: section_id=%s total_items=%s success=%s failed=%s cancelled=%s logic_nodes=%s references=%s",
            runtime.section.node_id,
            aggregated.stats["total_items"],
            aggregated.stats["success_count"],
            aggregated.stats["failed_count"],
            aggregated.stats["cancelled_count"],
            len(aggregated.logic_data_nodes),
            len(aggregated.section.reference_list),
        )
        return aggregated


@dataclass(slots=True)
class OpenAISectionItemStreamer:
    llm_client: OpenAILLMClient
    prompt_template: list[str]
    model: str = "gpt-4.1"
    temperature: float = 0
    lang: str = "zh"

    async def stream_items(
        self,
        section: SectionShell,
        image_base64: str,
        section_type: SectionType,
    ) -> AsyncIterator[str]:
        section_content = section.annotation or section.name
        logger.info(
            "Starting OpenAI section item stream: section_id=%s section_type=%s model=%s",
            section.node_id,
            section_type,
            self.model,
        )
        stream = self.llm_client.generate_result_by_llm(
            prompt_template=self.prompt_template,
            stream=True,
            model=self.model,
            temperature=self.temperature,
            lang=self.lang,
            section_type=section_type,
            section_content=section_content,
            image_base64=image_base64,
        )
        async for event in stream:
            logger.debug(
                "Forwarding streamed section item: section_id=%s index=%s raw_line=%s",
                section.node_id,
                event.index,
                event.raw_line,
            )
            yield json.dumps(event.object, ensure_ascii=False) + "\n"
        logger.info(
            "Completed OpenAI section item stream: section_id=%s section_type=%s",
            section.node_id,
            section_type,
        )


async def parse_section(
    section: SectionShell,
    image_base64: str,
    *,
    classifier: ClassifierFn,
    item_streamer: ItemStreamFn,
    max_concurrency: int = 5,
    fail_fast: bool = False,
    item_parser: JsonlIncrementalItemParser | None = None,
    aggregator: ParsedSectionAggregator | None = None,
    upstream_metadata: dict[str, Any] | None = None,
    page_no: int | None = None,
    section_bbox: tuple[float, float, float, float] | None = None,
    process_kv_item: KVItemHandlerFn | None = None,
    process_text_item: TextItemHandlerFn | None = None,
    process_col_item: ColItemHandlerFn | None = None,
    process_row_item: RowItemHandlerFn | None = None,
) -> ParsedSectionResult:
    logger.info(
        "Starting parse_section: section_id=%s fail_fast=%s max_concurrency=%s page_no=%s",
        section.node_id or upstream_metadata.get("section_node_id") if upstream_metadata else section.node_id,
        fail_fast,
        max_concurrency,
        page_no,
    )
    parser = item_parser or JsonlIncrementalItemParser()
    normalized_upstream_metadata = dict(upstream_metadata or {})
    _ensure_section_shell_defaults(section, normalized_upstream_metadata)
    scheduler = AsyncioItemTaskScheduler(
        max_concurrency=max_concurrency,
        fail_fast=fail_fast,
    )
    processor = DefaultItemProcessor(
        process_kv_item=process_kv_item or _default_kv_handler,
        process_text_item=process_text_item or _default_text_handler,
        process_col_item=process_col_item or _default_col_handler,
        process_row_item=process_row_item or _default_row_handler,
    )
    final_aggregator = aggregator or ParsedSectionAggregator()
    runtime = SectionRuntime(
        section=section,
        image_base64=image_base64,
        upstream_metadata=normalized_upstream_metadata,
        page_no=page_no,
        section_bbox=section_bbox,
    )

    section_type = _normalize_section_type(await _maybe_await(classifier(section, image_base64)))
    runtime.section.type = section_type
    runtime.classification_result["section_type"] = section_type
    logger.info(
        "Section classified: section_id=%s section_type=%s",
        runtime.section.node_id,
        section_type,
    )

    halted = False
    first_row_seen = False
    async for chunk in item_streamer(section, image_base64, section_type):
        logger.debug(
            "Received item stream chunk: section_id=%s chunk_length=%s",
            runtime.section.node_id,
            len(chunk),
        )
        for item in parser.feed(chunk):
            if not _validate_item_for_section(section_type, item):
                continue
            if (
                section_type == "table"
                and item.item_kind == "row"
                and not first_row_seen
            ):
                _sync_table_columns(runtime, await scheduler.wait_all())
                first_row_seen = True
            try:
                scheduler.add_item_task(item, runtime, processor)
            except SchedulerHaltedError:
                logger.warning(
                    "Scheduler halted while streaming items: section_id=%s item_id=%s",
                    runtime.section.node_id,
                    item.item_id,
                )
                halted = True
                break
        if halted:
            break

    if not halted:
        for item in parser.flush():
            if not _validate_item_for_section(section_type, item):
                continue
            if (
                section_type == "table"
                and item.item_kind == "row"
                and not first_row_seen
            ):
                _sync_table_columns(runtime, await scheduler.wait_all())
                first_row_seen = True
            try:
                scheduler.add_item_task(item, runtime, processor)
            except SchedulerHaltedError:
                logger.warning(
                    "Scheduler halted while flushing items: section_id=%s item_id=%s",
                    runtime.section.node_id,
                    item.item_id,
                )
                break

    item_results = await scheduler.wait_all()
    if section_type == "table":
        _sync_table_columns(runtime, item_results)
    if fail_fast and scheduler.has_failures:
        logger.error("parse_section failed in fail_fast mode: section_id=%s", runtime.section.node_id)
        raise SectionPipelineExecutionError(
            "Section pipeline failed in fail_fast mode due to item processing failure."
        )

    aggregated = await final_aggregator.aggregate(runtime, section_type, item_results)
    logger.info(
        "Completed parse_section: section_id=%s section_type=%s total_items=%s",
        runtime.section.node_id,
        section_type,
        aggregated.stats["total_items"],
    )
    return aggregated

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from llm_core.section_pipeline_impl import SectionShell, parse_section


def classify_section(section: SectionShell, image_base64: str) -> str:
    print(f"[A] classify section node_id={section.node_id!r}, image_size={len(image_base64)}")
    return "table"


async def stream_section_items(
    section: SectionShell,
    image_base64: str,
    section_type: str,
) -> AsyncIterator[str]:
    print(
        f"[B] start streaming, node_id={section.node_id!r}, "
        f"section_type={section_type}, image_size={len(image_base64)}"
    )
    lines = [
        {
            "item_id": "table-col-1",
            "item_kind": "col",
            "key": "总费用",
            "exp": "金额",
            "cbs_name": "total_fee",
            "is_summary": True,
            "order": 0,
        },
        {
            "item_id": "table-col-2",
            "item_kind": "col",
            "key": "药品费",
            "exp": "金额",
            "cbs_name": "drug_fee",
            "is_summary": False,
            "order": 1,
        },
        {
            "item_id": "table-row-1",
            "item_kind": "row",
            "key": "第一行",
            "annotation": "该行描述费用构成",
            "column_requirements": {
                "total_fee": "需要提取整行汇总金额",
                "drug_fee": "需要提取药品费金额",
            },
            "order": 2,
        },
    ]

    for line in lines:
        payload = json.dumps(line, ensure_ascii=False) + "\n"
        split_at = len(payload) // 2
        print(f"[B] emit chunk-1 for {line['item_id']}")
        yield payload[:split_at]
        await asyncio.sleep(0.08)
        print(f"[B] emit chunk-2 for {line['item_id']}")
        yield payload[split_at:]
        await asyncio.sleep(0.02)

    print("[B] streaming done")


async def main() -> None:
    section = SectionShell()
    image_base64 = "ZmFrZV9pbWFnZV9iYXNlNjQ="

    result = await parse_section(
        section,
        image_base64,
        classifier=classify_section,
        item_streamer=stream_section_items,
        max_concurrency=2,
        fail_fast=False,
        upstream_metadata={
            "source": "demo",
            "pdf_name": "demo.pdf",
            "section_node_id": "section-node-001",
        },
    )

    print("[D] parsed section result:")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

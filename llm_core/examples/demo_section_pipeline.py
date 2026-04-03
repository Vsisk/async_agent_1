from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from llm_core.section_pipeline_impl import SectionContext, parse_section


def classify_section(context: SectionContext) -> str:
    print(f"[A] classify section={context.section_id}")
    return "fee_table"


async def stream_section_items(
    context: SectionContext,
    section_type: str,
) -> AsyncIterator[str]:
    print(f"[B] start streaming, section_type={section_type}")
    lines = [
        {
            "item_id": "fee-1",
            "item_type": "fee_line",
            "raw_content": "挂号费 10 元",
            "order": 0,
        },
        {
            "item_id": "fee-2",
            "item_type": "fee_line",
            "raw_content": "检查费 200 元",
            "order": 1,
        },
        {
            "item_id": "fee-3",
            "item_type": "fee_line",
            "raw_content": "药品费 88 元",
            "order": 2,
        },
    ]

    for line in lines:
        payload = json.dumps(line, ensure_ascii=False) + "\n"
        split_at = len(payload) // 2
        chunk_1 = payload[:split_at]
        chunk_2 = payload[split_at:]
        print(f"[B] emit chunk-1 for {line['item_id']}")
        yield chunk_1
        await asyncio.sleep(0.08)
        print(f"[B] emit chunk-2 for {line['item_id']}")
        yield chunk_2
        await asyncio.sleep(0.02)

    print("[B] streaming done")


async def parse_item_content(item, context):
    print(f"[C] parse start item={item.item_id}")
    await asyncio.sleep(0.2)
    tokens = item.raw_content.split()
    result = {
        "raw": item.raw_content,
        "tokens": tokens,
        "section_type": context.section_type,
    }
    print(f"[C] parse done item={item.item_id}")
    return result


async def generate_item_description(item, parsed_result, context):
    print(f"[C] desc start item={item.item_id}")
    await asyncio.sleep(0.15)
    desc = f"{item.item_id} => {parsed_result['raw']}"
    print(f"[C] desc done item={item.item_id}")
    return desc


async def main() -> None:
    context = SectionContext(
        section_id="section-001",
        page_no=1,
        upstream_metadata={"source": "demo"},
    )

    node = await parse_section(
        context,
        classifier=classify_section,
        item_streamer=stream_section_items,
        parse_item_content=parse_item_content,
        generate_item_description=generate_item_description,
        max_concurrency=2,
        fail_fast=False,
    )

    print("[D] final structured node:")
    print(
        json.dumps(
            {
                "section_id": node.section_id,
                "section_type": node.section_type,
                "node_type": node.node_type,
                "fields": node.fields,
                "children": node.children,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

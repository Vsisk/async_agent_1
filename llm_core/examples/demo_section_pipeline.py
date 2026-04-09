from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict

from llm_core import OpenAILLMClient, OpenAISectionItemStreamer, SectionShell, parse_section


def classify_section(section: SectionShell, image_base64: str) -> str:
    print(f"[A] classify section node_id={section.node_id!r}, image_size={len(image_base64)}")
    return "table"


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run this demo.")

    section = SectionShell(
        annotation=(
            "项目一：总费用 10 元，药品费 8 元\n"
            "项目二：总费用 20 元，药品费 12 元"
        )
    )
    image_base64 = "ZmFrZV9pbWFnZV9iYXNlNjQ="

    llm_client = OpenAILLMClient()
    item_streamer = OpenAISectionItemStreamer(
        llm_client=llm_client,
        prompt_template=["SectionItemExtraction"],
        model="gpt-4.1",
        temperature=0,
        lang="zh",
    )

    result = await parse_section(
        section,
        image_base64,
        classifier=classify_section,
        item_streamer=item_streamer.stream_items,
        max_concurrency=2,
        fail_fast=False,
        upstream_metadata={
            "source": "demo",
            "pdf_name": "demo.pdf",
            "section_node_id": "section-node-001",
        },
    )

    print("[B] parsed section result:")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

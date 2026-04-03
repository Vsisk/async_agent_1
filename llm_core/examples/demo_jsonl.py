from __future__ import annotations

import asyncio

from llm_core import OpenAILLMClient


async def main() -> None:
    llm_client = OpenAILLMClient()

    async for item in llm_client.generate_result_by_llm(
        prompt_template=["SectionItemExtraction"],
        stream=True,
        model="gpt-4.1",
        temperature=0,
        section_type="fee_table",
        section_content="项目一：挂号费 10 元\n项目二：检查费 200 元\n项目三：药品费 88 元",
    ):
        print("index:", item.index, "object:", item.object)


if __name__ == "__main__":
    asyncio.run(main())

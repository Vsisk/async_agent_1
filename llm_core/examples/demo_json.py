from __future__ import annotations

import asyncio

from llm_core import OpenAILLMClient

MY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_select_type_modify",
        "schema": {
            "type": "object",
            "properties": {
                "section_type": {"type": "string"},
                "summary": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["section_type", "summary", "items"],
            "additionalProperties": False,
        },
    },
}


async def main() -> None:
    llm_client = OpenAILLMClient()

    response = await llm_client.generate_result_by_llm(
        prompt_template=["ImageSelectTypeModify"],
        response_format=MY_SCHEMA,
        stream=False,
        model="gpt-4.1",
        temperature=0,
        section_type="fee_table",
        raw_blocks="挂号费 10 元\n检查费 200 元\n药品费 88 元",
    )

    print("parsed:", response.parsed)
    print("raw_text:", response.raw_text)


if __name__ == "__main__":
    asyncio.run(main())

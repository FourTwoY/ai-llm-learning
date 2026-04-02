import json
import os
from pathlib import Path

from openai import OpenAI

try:
    from tools.basic_tools import FUNCTION_MAP, TOOLS
except ModuleNotFoundError:
    from qwen_agent_project.tools.basic_tools import FUNCTION_MAP, TOOLS


MODEL_NAME = "qwen3-max-2026-01-23"
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = [
    BASE_DIR / "qwen_agent_project" / ".env",
    BASE_DIR / "qwen_rag_project" / ".env",
]


def load_env_files() -> None:
    for env_path in ENV_CANDIDATES:
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_client() -> OpenAI:
    load_env_files()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "缺少环境变量 DASHSCOPE_API_KEY，请先在 qwen_agent_project/.env、"
            "qwen_rag_project/.env 或系统环境变量中配置后再运行 demo。"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def run_tool_selection_demo(user_query: str) -> None:
    client = build_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会调用工具的助手。"
                "如果用户在问当前时间，优先调用 get_current_time。"
                "如果用户要求查找本地文档、README、RAG 说明或 chunk 相关资料，调用 search_local_docs。"
                "如果工具已经返回结果，请基于工具结果给出简洁回答，不要假装自己查过本地文档。"
            ),
        },
        {"role": "user", "content": user_query},
    ]

    print("=" * 60)
    print(f"用户问题: {user_query}")
    print("=" * 60)

    first_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_msg = first_resp.choices[0].message
    tool_calls = assistant_msg.tool_calls or []

    if not tool_calls:
        print("模型未调用工具，直接回答：")
        print(assistant_msg.content)
        print("-" * 60)
        return

    messages.append(assistant_msg.model_dump())

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments or "{}"
        arguments = json.loads(arguments_str)

        if function_name not in FUNCTION_MAP:
            raise ValueError(f"未找到对应工具函数: {function_name}")

        print(f"选择工具: {function_name}")
        print(f"调用参数: {arguments_str}")

        tool_result = FUNCTION_MAP[function_name](**arguments)
        print("工具执行结果:")
        print(tool_result)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": tool_result,
            }
        )

    second_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    final_answer = second_resp.choices[0].message.content
    print("-" * 60)
    print("最终回答:")
    print(final_answer)
    print("-" * 60)


if __name__ == "__main__":
    demo_queries = [
        "现在几点了？",
        "帮我查一下本地文档里关于 RAG chunk 的说明",
    ]

    for query in demo_queries:
        run_tool_selection_demo(query)

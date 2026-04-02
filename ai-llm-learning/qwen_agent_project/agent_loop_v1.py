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


def run_agent(user_query: str, max_tool_rounds: int = 3) -> str:
    client = build_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会调用工具的助手。"
                "如果用户在问当前时间，优先调用 get_current_time。"
                "如果用户要求查找本地文档、README、RAG 说明或 chunk 相关资料，调用 search_local_docs。"
                "如果工具已经返回结果，请基于工具结果继续推进。"
                "如果已经足够回答，就直接给出最终答案，不要无意义重复调用工具。"
            ),
        },
        {"role": "user", "content": user_query},
    ]

    print("=" * 60)
    print(f"用户问题: {user_query}")
    print(f"最大工具轮数: {max_tool_rounds}")
    print("=" * 60)

    # 多步 agent 的核心就是这个循环：
    # 模型先思考是否需要工具；如果需要，就执行工具并把结果回填；
    # 然后继续让模型基于最新上下文再思考下一步。
    for round_index in range(1, max_tool_rounds + 1):
        print(f"[第 {round_index} 轮]")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls or []

        if not tool_calls:
            final_answer = assistant_msg.content or "模型未返回内容。"
            print("模型是否触发工具: 否")
            print("最终回答:")
            print(final_answer)
            print("-" * 60)
            return final_answer

        print("模型是否触发工具: 是")
        messages.append(assistant_msg.model_dump())

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments_str = tool_call.function.arguments or "{}"
            arguments = json.loads(arguments_str)

            print(f"触发工具: {function_name}")
            print(f"工具参数: {arguments_str}")

            if function_name not in FUNCTION_MAP:
                tool_result = f"工具执行失败：未找到工具 {function_name}"
            else:
                tool_result = FUNCTION_MAP[function_name](**arguments)

            print("工具结果:")
            print(tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        print("-" * 60)

    fallback_answer = (
        f"已达到最大工具调用轮数 {max_tool_rounds}，本轮先停止。"
        "如果需要，可以检查 tool description 或用户问题是否不够明确。"
    )
    print(fallback_answer)
    print("-" * 60)
    return fallback_answer


if __name__ == "__main__":
    demo_queries = [
        "现在几点了？",
        "帮我查一下本地文档里关于 RAG chunk 的说明",
    ]

    for query in demo_queries:
        run_agent(query)

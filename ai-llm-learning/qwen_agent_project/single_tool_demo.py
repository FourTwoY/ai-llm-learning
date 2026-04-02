import json
import os

from openai import OpenAI

from tools.basic_tools import TOOLS, FUNCTION_MAP


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("缺少环境变量 DASHSCOPE_API_KEY，请先在 .env 或系统环境变量中配置。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def run_single_tool_demo(user_query: str) -> None:
    client = build_client()

    messages = [
        {
            "role": "system",
            "content": "你是一个会调用工具的助手。遇到时间查询问题时，应优先调用工具，不要凭空编造时间。",
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    print("=" * 60)
    print(f"用户问题: {user_query}")
    print("=" * 60)

    # 第一次请求：让模型决定是否调用工具
    first_resp = client.chat.completions.create(
        model="qwen3-max-2026-01-23",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_msg = first_resp.choices[0].message
    tool_calls = assistant_msg.tool_calls

    # 情况1：模型没触发工具，直接回答
    if not tool_calls:
        print("模型未触发工具，直接回答：")
        print(assistant_msg.content)
        return

    print("模型决定调用工具。")
    # 把 assistant 的工具调用消息加入上下文
    messages.append(assistant_msg.model_dump())

    # 这里只处理一个工具调用，符合 Day 30 的“单工具 demo”
    tool_call = tool_calls[0]
    function_name = tool_call.function.name
    arguments_str = tool_call.function.arguments or "{}"

    print(f"调用工具名: {function_name}")
    print(f"调用参数: {arguments_str}")

    if function_name not in FUNCTION_MAP:
        raise ValueError(f"未找到对应工具函数: {function_name}")

    # 解析参数
    arguments = json.loads(arguments_str)

    # 真正执行工具
    tool_result = FUNCTION_MAP[function_name](**arguments)

    print(f"工具执行结果: {tool_result}")

    # 把工具结果回传给模型
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": tool_result,
        }
    )

    # 第二次请求：让模型基于工具结果给出最终自然语言回答
    second_resp = client.chat.completions.create(
        model="qwen3-max-2026-01-23",
        messages=messages,
    )

    final_answer = second_resp.choices[0].message.content

    print("-" * 60)
    print("最终回答:")
    print(final_answer)
    print("-" * 60)


if __name__ == "__main__":
    demo_query = "现在几点了？"
    run_single_tool_demo(demo_query)
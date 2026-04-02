from typing import Any

try:
    from agent_loop_v2 import (
        FUNCTION_MAP,
        MODEL_NAME,
        TOOLS,
        broken_demo_tool,
        build_client,
        empty_demo_tool,
        execute_tool_safely,
        format_tool_error,
        parse_tool_arguments,
    )
except ModuleNotFoundError:
    from qwen_agent_project.agent_loop_v2 import (
        FUNCTION_MAP,
        MODEL_NAME,
        TOOLS,
        broken_demo_tool,
        build_client,
        empty_demo_tool,
        execute_tool_safely,
        format_tool_error,
        parse_tool_arguments,
    )


def build_output_summary(tool_result: str, limit: int = 120) -> str:
    normalized = " ".join(tool_result.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def infer_tool_success(tool_result: str) -> tuple[bool, str | None]:
    if "执行失败" in tool_result:
        lines = [line.strip() for line in tool_result.splitlines() if line.strip()]
        if len(lines) >= 2:
            return False, lines[1]
        return False, tool_result

    if "返回了空结果" in tool_result:
        return False, "工具返回空结果。"

    return True, None


def run_agent_with_trace(
    user_query: str,
    max_tool_rounds: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    tool_call_trace: list[dict[str, Any]] = []
    tool_outputs_summary: list[dict[str, Any]] = []

    try:
        client = build_client()
    except Exception as exc:  # noqa: BLE001
        return {
            "final_answer": f"初始化失败：{exc}",
            "tool_call_trace": tool_call_trace,
            "tool_outputs_summary": tool_outputs_summary,
        }

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会调用工具的助手。"
                "如果用户在问当前时间，优先调用 get_current_time。"
                "如果用户要求查找本地文档、README、RAG 说明或 chunk 相关资料，调用 search_local_docs。"
                "如果工具执行失败，请读取 tool message 中的错误并继续给出可读回复。"
                "如果工具返回空结果，请明确告诉用户没有找到内容。"
                "如果已经足够回答，就直接给出最终答案。"
            ),
        },
        {"role": "user", "content": user_query},
    ]

    for round_index in range(1, max_tool_rounds + 1):
        if verbose:
            print(f"[API 第 {round_index} 轮] {user_query}")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "final_answer": f"模型调用失败：{type(exc).__name__}: {exc}",
                "tool_call_trace": tool_call_trace,
                "tool_outputs_summary": tool_outputs_summary,
            }

        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls or []

        if not tool_calls:
            return {
                "final_answer": assistant_msg.content or "模型未返回内容。",
                "tool_call_trace": tool_call_trace,
                "tool_outputs_summary": tool_outputs_summary,
            }

        messages.append(assistant_msg.model_dump())

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments_str = tool_call.function.arguments or "{}"

            try:
                arguments = parse_tool_arguments(arguments_str)
                tool_result = execute_tool_safely(function_name, arguments)
            except Exception as exc:  # noqa: BLE001
                arguments = {}
                tool_result = format_tool_error(
                    function_name,
                    f"主循环捕获到未预期异常：{type(exc).__name__}: {exc}",
                )

            success, error_message = infer_tool_success(tool_result)
            tool_call_trace.append(
                {
                    "round": round_index,
                    "tool_name": function_name,
                    "tool_args": arguments,
                    "success": success,
                    "error_message": error_message,
                }
            )
            tool_outputs_summary.append(
                {
                    "round": round_index,
                    "tool_name": function_name,
                    "summary": build_output_summary(tool_result),
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

    return {
        "final_answer": (
            f"已达到最大工具调用轮数 {max_tool_rounds}，本轮先停止。"
            " 如果还需要继续，可以补充参数或缩小问题范围后重试。"
        ),
        "tool_call_trace": tool_call_trace,
        "tool_outputs_summary": tool_outputs_summary,
    }


def run_debug_tool_case(case_name: str) -> dict[str, Any]:
    case_map: dict[str, tuple[str, dict[str, Any], Any]] = {
        "empty_args": ("search_local_docs", {"query": ""}, None),
        "internal_error": ("broken_demo_tool", {"city": "Shanghai"}, broken_demo_tool),
        "empty_result": ("empty_demo_tool", {"keyword": "RAG"}, empty_demo_tool),
    }

    if case_name not in case_map:
        return {
            "final_answer": (
                "未识别的 debug_tool_case。"
                " 可选值：empty_args、internal_error、empty_result。"
            ),
            "tool_call_trace": [],
            "tool_outputs_summary": [],
        }

    function_name, arguments, injected_function = case_map[case_name]
    original_function_map = dict(FUNCTION_MAP)

    try:
        if injected_function is not None:
            FUNCTION_MAP[function_name] = injected_function
        tool_result = execute_tool_safely(function_name, arguments)
    finally:
        FUNCTION_MAP.clear()
        FUNCTION_MAP.update(original_function_map)

    success, error_message = infer_tool_success(tool_result)
    return {
        "final_answer": (
            "这是一个调试模式下的工具异常演示结果。"
            " 接口已经成功返回可读错误，没有因为工具异常而崩溃。"
        ),
        "tool_call_trace": [
            {
                "round": 1,
                "tool_name": function_name,
                "tool_args": arguments,
                "success": success,
                "error_message": error_message,
            }
        ],
        "tool_outputs_summary": [
            {
                "round": 1,
                "tool_name": function_name,
                "summary": build_output_summary(tool_result),
            }
        ],
    }

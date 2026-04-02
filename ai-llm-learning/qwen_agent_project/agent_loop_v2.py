import json
import os
from pathlib import Path
from typing import Any

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


class ToolExecutionError(Exception):
    """可读的工具执行错误，供主循环安全消费。"""


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
            "qwen_rag_project/.env 或系统环境变量中配置后再运行。"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def build_tool_schema_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schema_map: dict[str, dict[str, Any]] = {}
    for item in tools:
        function_block = item.get("function", {})
        name = function_block.get("name")
        if name:
            schema_map[name] = function_block
    return schema_map


TOOL_SCHEMA_MAP = build_tool_schema_map(TOOLS)


def parse_tool_arguments(arguments_str: str | None) -> dict[str, Any]:
    if not arguments_str:
        return {}

    try:
        parsed = json.loads(arguments_str)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"工具参数不是合法 JSON：{exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ToolExecutionError("工具参数格式错误：参数必须是 JSON object。")

    return parsed


def validate_tool_arguments(function_name: str, arguments: dict[str, Any]) -> None:
    schema = TOOL_SCHEMA_MAP.get(function_name, {})
    parameters = schema.get("parameters", {})
    required_fields = parameters.get("required", [])

    for field in required_fields:
        value = arguments.get(field)
        if value is None:
            raise ToolExecutionError(f"工具参数缺失：`{field}` 不能为空。")
        if isinstance(value, str) and not value.strip():
            raise ToolExecutionError(f"工具参数无效：`{field}` 不能为空字符串。")


def is_empty_tool_result(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        return not result.strip()
    if isinstance(result, (list, dict, tuple, set)):
        return len(result) == 0
    return False


def format_tool_error(function_name: str, error_message: str) -> str:
    return (
        f"工具 `{function_name}` 执行失败。\n"
        f"错误原因：{error_message}\n"
        "请根据这个错误调整参数，或改用别的方式回答用户。"
    )


def execute_tool_safely(function_name: str, arguments: dict[str, Any]) -> str:
    if function_name not in FUNCTION_MAP:
        return format_tool_error(function_name, f"未找到名为 `{function_name}` 的工具。")

    try:
        validate_tool_arguments(function_name, arguments)
        result = FUNCTION_MAP[function_name](**arguments)
    except TypeError as exc:
        return format_tool_error(function_name, f"参数不匹配：{exc}")
    except ToolExecutionError as exc:
        return format_tool_error(function_name, str(exc))
    except Exception as exc:  # noqa: BLE001
        return format_tool_error(function_name, f"工具内部异常：{type(exc).__name__}: {exc}")

    if is_empty_tool_result(result):
        return (
            f"工具 `{function_name}` 执行完成，但返回了空结果。"
            " 请不要把它当成正常命中，建议换个关键词、补充上下文，或直接说明未找到信息。"
        )

    return str(result)


def run_agent(user_query: str, max_tool_rounds: int = 3) -> str:
    try:
        client = build_client()
    except Exception as exc:  # noqa: BLE001
        return f"初始化失败：{exc}"

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会调用工具的助手。"
                "如果用户在问当前时间，优先调用 get_current_time。"
                "如果用户要求查找本地文档、README、RAG 说明或 chunk 相关资料，调用 search_local_docs。"
                "如果工具执行失败，请读取 tool message 中的错误并继续给出可读回复，不要让回答中断。"
                "如果工具返回空结果，请明确告诉用户没有找到内容，并给出下一步建议。"
                "如果已经足够回答，就直接给出最终答案，不要无意义重复调用工具。"
            ),
        },
        {"role": "user", "content": user_query},
    ]

    print("=" * 60)
    print(f"用户问题: {user_query}")
    print(f"最大工具轮数: {max_tool_rounds}")
    print("=" * 60)

    for round_index in range(1, max_tool_rounds + 1):
        print(f"[第 {round_index} 轮]")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",  # 模型自主判断是否调用工具
            )
        except Exception as exc:  # noqa: BLE001
            error_message = f"模型调用失败：{type(exc).__name__}: {exc}"
            print(error_message)
            print("-" * 60)
            return error_message

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

            print(f"触发工具: {function_name}")
            print(f"工具参数: {arguments_str}")

            try:
                arguments = parse_tool_arguments(arguments_str)
                tool_result = execute_tool_safely(function_name, arguments)
            except Exception as exc:  # noqa: BLE001
                tool_result = format_tool_error(
                    function_name,
                    f"主循环捕获到未预期异常：{type(exc).__name__}: {exc}",
                )

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
        " 如果还需要继续，可以检查工具描述、补充参数，或缩小问题范围后重试。"
    )
    print(fallback_answer)
    print("-" * 60)
    return fallback_answer


def broken_demo_tool(city: str) -> str:
    raise RuntimeError(f"模拟内部异常：城市服务暂时不可用，city={city}")


def empty_demo_tool(keyword: str) -> str:
    _ = keyword
    return ""


def simulate_tool_error_cases() -> list[dict[str, str]]:
    demo_function_map = dict(FUNCTION_MAP)
    demo_function_map["broken_demo_tool"] = broken_demo_tool
    demo_function_map["empty_demo_tool"] = empty_demo_tool

    cases = [
        {
            "case": "工具参数为空",
            "function_name": "search_local_docs",
            "arguments": {"query": ""},
        },
        {
            "case": "工具内部报错",
            "function_name": "broken_demo_tool",
            "arguments": {"city": "Shanghai"},
        },
        {
            "case": "工具返回空结果",
            "function_name": "empty_demo_tool",
            "arguments": {"keyword": "RAG"},
        },
    ]

    results: list[dict[str, str]] = []
    for item in cases:
        function_name = item["function_name"]
        arguments = item["arguments"]

        original_function_map = dict(FUNCTION_MAP)
        try:
            FUNCTION_MAP.update(demo_function_map)
            result = execute_tool_safely(function_name, arguments)
        finally:
            FUNCTION_MAP.clear()
            FUNCTION_MAP.update(original_function_map)

        results.append(
            {
                "case": item["case"],
                "function_name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
                "result": result,
            }
        )

    return results


if __name__ == "__main__":
    print("=== 工具异常场景模拟 ===")
    for item in simulate_tool_error_cases():
        print(f"[场景] {item['case']}")
        print(f"工具: {item['function_name']}")
        print(f"参数: {item['arguments']}")
        print("结果:")
        print(item["result"])
        print("-" * 60)

    demo_queries = [
        "现在几点了？",
        "帮我查一下本地文档里关于 RAG chunk 的说明",
    ]

    for query in demo_queries:
        run_agent(query)

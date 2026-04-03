import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from tools.basic_tools import get_current_time, search_local_docs
except ModuleNotFoundError:
    from qwen_agent_project.tools.basic_tools import get_current_time, search_local_docs


RouteName = Literal["chat", "retrieval", "tool"]


class RouterState(TypedDict):
    # Day 38 的最小 state，只保留 4 个字段
    user_question: str
    route: str
    tool_result: str
    final_answer: str


CHAT_KEYWORDS = ("你好", "您好", "hi", "hello", "在吗", "嗨", "哈喽")
TIME_KEYWORDS = ("几点", "时间", "现在几点", "当前时间")
CALC_KEYWORDS = ("计算", "加", "减", "乘", "除", "+", "-", "*", "/", "等于")


def detect_route(question: str) -> RouteName:
    normalized = question.lower().strip()

    if any(keyword in normalized for keyword in CHAT_KEYWORDS):
        return "chat"

    if any(keyword in normalized for keyword in TIME_KEYWORDS + CALC_KEYWORDS):
        return "tool"

    return "retrieval"


def safe_calculate(expression: str) -> str:
    # 只做最小教学 demo：支持数字、小数点、空格和四则运算符
    matched = re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", expression)
    candidate = "".join(matched).strip()

    if not candidate:
        return "计算工具未提取到可执行表达式。"

    try:
        result = eval(candidate, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"计算工具执行失败：{type(exc).__name__}: {exc}"

    return f"{candidate} = {result}"


def router(state: RouterState) -> RouterState:
    route = detect_route(state["user_question"])
    return {
        **state,
        "route": route,
    }


def direct_respond(state: RouterState) -> RouterState:
    return {
        **state,
        "tool_result": "未调用工具",
        "final_answer": "你好，我在。这个问题属于简单闲聊，所以这次我直接回答，没有调用工具。",
    }


def call_retrieval_tool(state: RouterState) -> RouterState:
    tool_result = search_local_docs(state["user_question"])
    return {
        **state,
        "tool_result": tool_result,
        "final_answer": f"这是检索分支的结果：\n{tool_result}",
    }


def call_calc_or_time_tool(state: RouterState) -> RouterState:
    question = state["user_question"]

    if any(keyword in question.lower() for keyword in TIME_KEYWORDS):
        tool_result = f"当前时间：{get_current_time()}"
        final_answer = f"这是时间工具分支的结果：{tool_result}"
    else:
        tool_result = safe_calculate(question)
        final_answer = f"这是计算工具分支的结果：{tool_result}"

    return {
        **state,
        "tool_result": tool_result,
        "final_answer": final_answer,
    }


def choose_next_node(state: RouterState) -> str:
    route = state["route"]
    if route == "chat":
        return "direct_respond"
    if route == "tool":
        return "call_calc_or_time_tool"
    return "call_retrieval_tool"


def build_graph():
    # graph: 描述 state 在哪些 node 之间流动
    graph_builder = StateGraph(RouterState)

    # node: 图里的处理步骤
    graph_builder.add_node("router", router)
    graph_builder.add_node("direct_respond", direct_respond)
    graph_builder.add_node("call_retrieval_tool", call_retrieval_tool)
    graph_builder.add_node("call_calc_or_time_tool", call_calc_or_time_tool)

    # edge: 从 START 进入 router，然后根据 route 分流到不同分支
    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges(
        "router",
        choose_next_node,
        {
            "direct_respond": "direct_respond",
            "call_retrieval_tool": "call_retrieval_tool",
            "call_calc_or_time_tool": "call_calc_or_time_tool",
        },
    )
    graph_builder.add_edge("direct_respond", END)
    graph_builder.add_edge("call_retrieval_tool", END)
    graph_builder.add_edge("call_calc_or_time_tool", END)

    # compile: 把图编译成可执行对象
    return graph_builder.compile()


def run_demo(question: str) -> None:
    graph = build_graph()
    initial_state: RouterState = {
        "user_question": question,
        "route": "",
        "tool_result": "",
        "final_answer": "",
    }
    final_state = graph.invoke(initial_state)

    print("=" * 60)
    print("user_question:")
    print(final_state["user_question"])
    print("-" * 60)
    print("route:")
    print(final_state["route"])
    print("-" * 60)
    print("tool_result:")
    print(final_state["tool_result"])
    print("-" * 60)
    print("final_answer:")
    print(final_state["final_answer"])
    print("=" * 60)


if __name__ == "__main__":
    demo_questions = [
        "你好，在吗？",
        "帮我查一下本地文档里关于 RAG chunk 的说明",
        "现在几点了？",
    ]

    for demo_question in demo_questions:
        run_demo(demo_question)

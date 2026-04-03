from typing import TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from agent_loop_v2 import MODEL_NAME, build_client
    from tools.basic_tools import search_local_docs
except ModuleNotFoundError:
    from qwen_agent_project.agent_loop_v2 import MODEL_NAME, build_client
    from qwen_agent_project.tools.basic_tools import search_local_docs


class GraphState(TypedDict):
    # LangGraph 里流动的 state，只保留 Day 37 需要的 4 个字段
    user_question: str
    rewritten_query: str
    tool_result: str
    final_answer: str


def decide(state: GraphState) -> GraphState:
    """node 1: 先让模型把用户问题改写成更适合工具使用的查询词。"""
    client = build_client()
    prompt = (
        "请把用户问题改写成更适合本地文档检索工具使用的一句话查询。"
        "只返回改写结果，不要解释。"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": state["user_question"]},
        ],
    )
    rewritten_query = response.choices[0].message.content or state["user_question"]

    return {
        **state,
        "rewritten_query": rewritten_query.strip(),
    }


def call_tool(state: GraphState) -> GraphState:
    """node 2: 调用一个最简单的现有工具，这里复用 search_local_docs。"""
    tool_result = search_local_docs(state["rewritten_query"])
    return {
        **state,
        "tool_result": tool_result,
    }


def respond(state: GraphState) -> GraphState:
    """node 3: 基于用户原问题 + 改写结果 + 工具结果生成最终回答。"""
    client = build_client()
    prompt = (
        "你是一个问答助手。"
        "请基于用户问题、改写后的查询、以及工具结果，输出简洁清楚的最终回答。"
        "如果工具结果没有找到内容，也要明确说出来。"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"用户问题: {state['user_question']}\n"
                    f"改写后的查询: {state['rewritten_query']}\n"
                    f"工具结果: {state['tool_result']}"
                ),
            },
        ],
    )
    final_answer = response.choices[0].message.content or "模型未返回最终回答。"

    return {
        **state,
        "final_answer": final_answer.strip(),
    }


def build_graph():
    # graph: 定义 state 的结构
    graph_builder = StateGraph(GraphState)

    # node: 图中的处理步骤
    graph_builder.add_node("decide", decide)
    graph_builder.add_node("call_tool", call_tool)
    graph_builder.add_node("respond", respond)

    # edge: 固定执行路径 START -> decide -> call_tool -> respond -> END
    graph_builder.add_edge(START, "decide")
    graph_builder.add_edge("decide", "call_tool")
    graph_builder.add_edge("call_tool", "respond")
    graph_builder.add_edge("respond", END)

    # compile: 把定义好的图编译成可执行对象
    return graph_builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    initial_state: GraphState = {
        "user_question": "帮我查一下本地文档里关于 RAG chunk 的说明",
        "rewritten_query": "",
        "tool_result": "",
        "final_answer": "",
    }

    # invoke: 执行整张图
    final_state = graph.invoke(initial_state)

    print("=" * 60)
    print("user_question:")
    print(final_state["user_question"])
    print("-" * 60)
    print("rewritten_query:")
    print(final_state["rewritten_query"])
    print("-" * 60)
    print("tool_result:")
    print(final_state["tool_result"])
    print("-" * 60)
    print("final_answer:")
    print(final_state["final_answer"])
    print("=" * 60)

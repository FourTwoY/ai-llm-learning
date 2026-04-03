import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RAG_PROJECT_DIR = PROJECT_ROOT / "qwen_rag_project"
STATE_TRACE_PATH = CURRENT_DIR / "state_trace_example.json"

for import_path in (str(RAG_PROJECT_DIR), str(PROJECT_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

# qwen_rag_project 的配置文件里使用了 data/... 相对路径，
# 这里先切换工作目录，避免从 qwen_agent_project 启动时找不到 chunks/embeddings。
os.chdir(RAG_PROJECT_DIR)


from qwen_rag_project.config import get_config  # noqa: E402
from qwen_rag_project.main import ensure_embeddings_ready  # noqa: E402
from qwen_rag_project.services.embedding_service import load_embeddings  # noqa: E402
from qwen_rag_project.services.generation_service import generate_answer  # noqa: E402
from qwen_rag_project.services.query_rewrite_service import rewrite_query  # noqa: E402
from qwen_rag_project.services.retrieval_service import retrieve_chunks  # noqa: E402

try:
    from tools.basic_tools import get_current_time
except ModuleNotFoundError:
    from qwen_agent_project.tools.basic_tools import get_current_time


RouteName = Literal["chat", "tool", "retrieval"]


class RewriteGraphState(TypedDict):
    original_question: str
    rewritten_query: str
    used_tools: list[str]
    tool_outputs: dict[str, Any]
    final_answer: str
    route: str
    retrieved_chunks: list[dict]
    top_k: int


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


def serialize_chunks(chunks: list[dict]) -> list[dict]:
    serialized = []
    for item in chunks:
        serialized.append(
            {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "score": float(item.get("score", 0.0)),
                "text": item.get("text", ""),
            }
        )
    return serialized


def append_tool(state: RewriteGraphState, tool_name: str, output: Any) -> RewriteGraphState:
    used_tools = list(state["used_tools"])
    tool_outputs = dict(state["tool_outputs"])

    used_tools.append(tool_name)
    tool_outputs[tool_name] = output

    return {
        **state,
        "used_tools": used_tools,
        "tool_outputs": tool_outputs,
    }


def safe_calculate(expression: str) -> str:
    candidate = "".join(re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", expression)).strip()
    if not candidate:
        return "计算工具未提取到可执行表达式。"

    try:
        result = eval(candidate, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"计算工具执行失败：{type(exc).__name__}: {exc}"
    return f"{candidate} = {result}"


def classify(state: RewriteGraphState) -> RewriteGraphState:
    route = detect_route(state["original_question"])
    next_state = {
        **state,
        "route": route,
    }
    return append_tool(next_state, "classify", {"route": route})


def rewrite_query_node(state: RewriteGraphState) -> RewriteGraphState:
    rewritten = rewrite_query(state["original_question"])
    next_state = {
        **state,
        "rewritten_query": rewritten,
    }
    return append_tool(next_state, "rewrite_query", {"rewritten_query": rewritten})


def retrieve_knowledge(state: RewriteGraphState) -> RewriteGraphState:
    ensure_embeddings_ready()
    cfg = get_config()
    embedded_chunks = load_embeddings(cfg["paths"]["embeddings_file"])
    query_for_retrieval = state["rewritten_query"] or state["original_question"]
    retrieved_chunks = retrieve_chunks(
        query_for_retrieval,
        embedded_chunks,
        top_k=state["top_k"],
    )
    serialized_chunks = serialize_chunks(retrieved_chunks)

    next_state = {
        **state,
        "retrieved_chunks": serialized_chunks,
    }
    return append_tool(
        next_state,
        "retrieve_chunks",
        {
            "query": query_for_retrieval,
            "top_k": state["top_k"],
            "chunks": serialized_chunks,
        },
    )


def answer(state: RewriteGraphState) -> RewriteGraphState:
    if state["route"] == "chat":
        final_answer = "这是直接回答分支：当前问题属于简单闲聊，所以没有进入 rewrite 和 RAG 检索。"
        next_state = {**state, "final_answer": final_answer}
        return append_tool(next_state, "direct_answer", {"answer": final_answer})

    if state["route"] == "tool":
        question = state["original_question"].lower()
        if any(keyword in question for keyword in TIME_KEYWORDS):
            tool_output = f"当前时间：{get_current_time()}"
        else:
            tool_output = safe_calculate(state["original_question"])

        final_answer = f"这是工具分支的直接回答：{tool_output}"
        next_state = {**state, "final_answer": final_answer}
        return append_tool(next_state, "direct_tool_answer", {"answer": final_answer})

    if not state["retrieved_chunks"]:
        final_answer = "rewrite 和 retrieve 已执行，但没有检索到可用 chunks，所以暂时无法生成可靠回答。"
        next_state = {**state, "final_answer": final_answer}
        return append_tool(next_state, "generate_answer", {"answer": final_answer, "fallback": True})

    final_answer = generate_answer(state["original_question"], state["retrieved_chunks"])
    next_state = {**state, "final_answer": final_answer}
    return append_tool(next_state, "generate_answer", {"answer": final_answer})


def choose_after_classify(state: RewriteGraphState) -> str:
    if state["route"] == "retrieval":
        return "rewrite_query_node"
    return "answer"


def build_graph():
    graph_builder = StateGraph(RewriteGraphState)
    graph_builder.add_node("classify", classify)
    graph_builder.add_node("rewrite_query_node", rewrite_query_node)
    graph_builder.add_node("retrieve_knowledge", retrieve_knowledge)
    graph_builder.add_node("answer", answer)

    graph_builder.add_edge(START, "classify")
    graph_builder.add_conditional_edges(
        "classify",
        choose_after_classify,
        {
            "rewrite_query_node": "rewrite_query_node",
            "answer": "answer",
        },
    )
    graph_builder.add_edge("rewrite_query_node", "retrieve_knowledge")
    graph_builder.add_edge("retrieve_knowledge", "answer")
    graph_builder.add_edge("answer", END)
    return graph_builder.compile()


def run_retrieval_only(question: str, top_k: int = 3) -> dict[str, Any]:
    ensure_embeddings_ready()
    cfg = get_config()
    embedded_chunks = load_embeddings(cfg["paths"]["embeddings_file"])
    chunks = retrieve_chunks(question, embedded_chunks, top_k=top_k)
    serialized_chunks = serialize_chunks(chunks)

    if serialized_chunks:
        final_answer = generate_answer(question, serialized_chunks)
    else:
        final_answer = "未 rewrite 的检索没有命中可用 chunks。"

    return {
        "rewritten_query": question,
        "retrieved_chunks": serialized_chunks,
        "final_answer": final_answer,
    }


def compare_with_and_without_rewrite(question: str, top_k: int = 3) -> dict[str, Any]:
    without_rewrite = run_retrieval_only(question, top_k=top_k)

    graph = build_graph()
    with_rewrite_state: RewriteGraphState = {
        "original_question": question,
        "rewritten_query": "",
        "used_tools": [],
        "tool_outputs": {},
        "final_answer": "",
        "route": "",
        "retrieved_chunks": [],
        "top_k": top_k,
    }
    with_rewrite = graph.invoke(with_rewrite_state)

    return {
        "question": question,
        "without_rewrite": without_rewrite,
        "with_rewrite": {
            "rewritten_query": with_rewrite["rewritten_query"],
            "retrieved_chunks": with_rewrite["retrieved_chunks"],
            "final_answer": with_rewrite["final_answer"],
        },
    }


def write_state_trace(question: str, top_k: int = 3) -> dict[str, Any]:
    graph = build_graph()
    initial_state: RewriteGraphState = {
        "original_question": question,
        "rewritten_query": "",
        "used_tools": [],
        "tool_outputs": {},
        "final_answer": "",
        "route": "",
        "retrieved_chunks": [],
        "top_k": top_k,
    }

    snapshots = []
    for state in graph.stream(initial_state, stream_mode="values"):
        snapshots.append(state)

    payload = {
        "question": question,
        "snapshots": snapshots,
        "final_state": snapshots[-1] if snapshots else initial_state,
    }
    STATE_TRACE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    demo_question = "帮我查一下本地文档里关于 RAG  的说明"

    comparison = compare_with_and_without_rewrite(demo_question, top_k=3)
    trace_payload = write_state_trace(demo_question, top_k=3)

    print("=" * 60)
    print("original_question:")
    print(demo_question)
    print("-" * 60)
    print("without rewrite -> rewritten_query:")
    print(comparison["without_rewrite"]["rewritten_query"])
    print("without rewrite -> retrieved_chunks:")
    print(json.dumps(comparison["without_rewrite"]["retrieved_chunks"], ensure_ascii=False, indent=2))
    print("without rewrite -> final_answer:")
    print(comparison["without_rewrite"]["final_answer"])
    print("-" * 60)
    print("with rewrite -> rewritten_query:")
    print(comparison["with_rewrite"]["rewritten_query"])
    print("with rewrite -> retrieved_chunks:")
    print(json.dumps(comparison["with_rewrite"]["retrieved_chunks"], ensure_ascii=False, indent=2))
    print("with rewrite -> final_answer:")
    print(comparison["with_rewrite"]["final_answer"])
    print("-" * 60)
    print("state_trace_example.json:")
    print(STATE_TRACE_PATH)
    print("final_state:")
    print(json.dumps(trace_payload["final_state"], ensure_ascii=False, indent=2))
    print("=" * 60)

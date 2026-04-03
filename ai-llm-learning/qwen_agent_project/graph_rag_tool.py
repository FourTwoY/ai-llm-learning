import json
import os
import re
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RAG_PROJECT_DIR = PROJECT_ROOT / "qwen_rag_project"

for import_path in (str(RAG_PROJECT_DIR), str(PROJECT_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

# qwen_rag_project 里的 chunks/embeddings 路径是相对 data/... 配的，
# 切到 RAG 项目目录后再调用 ensure_embeddings_ready/load_embeddings 才不会找错文件。
os.chdir(RAG_PROJECT_DIR)


from qwen_rag_project.config import get_config  # noqa: E402
from qwen_rag_project.main import ensure_embeddings_ready  # noqa: E402
from qwen_rag_project.services.embedding_service import load_embeddings  # noqa: E402
from qwen_rag_project.services.generation_service import generate_answer  # noqa: E402
from qwen_rag_project.services.retrieval_service import retrieve_chunks  # noqa: E402

try:
    from tools.basic_tools import get_current_time
except ModuleNotFoundError:
    from qwen_agent_project.tools.basic_tools import get_current_time


RouteName = Literal["retrieval", "tool", "chat"]


class RagGraphState(TypedDict):
    query: str
    route: str
    top_k: int
    tool_result: list[dict]
    final_answer: str


TIME_KEYWORDS = ("几点", "时间", "现在几点", "当前时间")
CALC_KEYWORDS = ("计算", "加", "减", "乘", "除", "+", "-", "*", "/", "等于")
CHAT_KEYWORDS = ("你好", "您好", "hi", "hello", "在吗", "嗨", "哈喽")


def detect_route(query: str) -> RouteName:
    normalized = query.lower().strip()
    if any(keyword in normalized for keyword in CHAT_KEYWORDS):
        return "chat"
    if any(keyword in normalized for keyword in TIME_KEYWORDS + CALC_KEYWORDS):
        return "tool"
    return "retrieval"


def safe_calculate(expression: str) -> str:
    candidate = "".join(re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", expression)).strip()
    if not candidate:
        return "计算工具未提取到可执行表达式。"

    try:
        result = eval(candidate, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"计算工具执行失败：{type(exc).__name__}: {exc}"
    return f"{candidate} = {result}"


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


def classify(state: RagGraphState) -> RagGraphState:
    return {
        **state,
        "route": detect_route(state["query"]),
    }


def retrieve_knowledge(state: RagGraphState) -> RagGraphState:
    ensure_embeddings_ready()
    cfg = get_config()
    embedded_chunks = load_embeddings(cfg["paths"]["embeddings_file"])
    retrieved_chunks = retrieve_chunks(
        state["query"],
        embedded_chunks,
        top_k=state["top_k"],
    )
    return {
        **state,
        "tool_result": serialize_chunks(retrieved_chunks),
    }


def answer(state: RagGraphState) -> RagGraphState:
    if state["route"] == "chat":
        final_answer = "这是简单问候，所以这里直接回答，不调用检索工具。"
        return {**state, "final_answer": final_answer}

    if state["route"] == "tool":
        question = state["query"].lower()
        if any(keyword in question for keyword in TIME_KEYWORDS):
            final_answer = f"这是时间工具分支的结果：当前时间：{get_current_time()}"
        else:
            final_answer = f"这是计算工具分支的结果：{safe_calculate(state['query'])}"
        return {**state, "final_answer": final_answer}

    if not state["tool_result"]:
        return {
            **state,
            "final_answer": "检索分支已执行，但没有拿到可用 chunks，所以暂时无法生成回答。",
        }

    final_answer = generate_answer(state["query"], state["tool_result"])
    return {
        **state,
        "final_answer": final_answer,
    }


def choose_next_node(state: RagGraphState) -> str:
    if state["route"] == "retrieval":
        return "retrieve_knowledge"
    return "answer"


def build_graph():
    graph_builder = StateGraph(RagGraphState)
    graph_builder.add_node("classify", classify)
    graph_builder.add_node("retrieve_knowledge", retrieve_knowledge)
    graph_builder.add_node("answer", answer)

    graph_builder.add_edge(START, "classify")
    graph_builder.add_conditional_edges(
        "classify",
        choose_next_node,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "answer": "answer",
        },
    )
    graph_builder.add_edge("retrieve_knowledge", "answer")
    graph_builder.add_edge("answer", END)
    return graph_builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    initial_state: RagGraphState = {
        "query": "帮我查一下本地文档里关于 RAG chunk 的说明",
        "route": "",
        "top_k": 3,
        "tool_result": [],
        "final_answer": "",
    }

    final_state = graph.invoke(initial_state)

    print("=" * 60)
    print("query:")
    print(final_state["query"])
    print("-" * 60)
    print("top-k chunks:")
    print(json.dumps(final_state["tool_result"], ensure_ascii=False, indent=2))
    print("-" * 60)
    print("final_answer:")
    print(final_state["final_answer"])
    print("=" * 60)

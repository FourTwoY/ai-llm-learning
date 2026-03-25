import os
from pathlib import Path
from openai import OpenAI

from ..services.embedding_service import load_embeddings
from ..services.retrieval_service import retrieve_chunks
from ..services.rerank_service import rerank_chunks
from ..services.generation_service import generate_answer


EMBEDDINGS_FILE = "qwen_rag_project/data/embeddings/all_embeddings.json"
CHAT_MODEL = "qwen3-max-2026-01-23"
OUTPUT_FILE = "qwen_rag_project/evaluation/rag_eval.md"


EVAL_QUESTIONS = [
    "Vision Transformer 的核心思想是什么？",
    "ViT 和 CNN 的主要区别是什么？",
    "这篇论文为什么要把图像切成 patch？",
    "ViT 的输入是如何构造的？",
    "Class Token 在 ViT 里起什么作用？",
    "这篇论文的关键创新点有哪些？",
    "ViT 在什么条件下能表现得比 CNN 更好？",
    "这篇论文主要解决了什么问题？",
    "ViT 的方法有什么局限性或前提条件？",
    "如果让我向初学者解释 ViT，这篇论文最值得记住的一点是什么？",
]


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def ask_directly(query: str) -> str:
    """
    方案1：不做 RAG，直接问千问
    """
    client = get_client()

    system_prompt = """
你是一个学术问答助手。
请直接回答用户问题。
如果不确定，请明确说明不确定，不要胡编乱造。
""".strip()

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content


def rag_without_rerank(query: str, embedded_chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    方案2：embedding 检索 + 生成
    """
    retrieved = retrieve_chunks(query, embedded_chunks, top_k=3)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


def rag_with_rerank(query: str, embedded_chunks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """
    方案3：embedding 检索 + rerank + 生成
    """
    retrieved = retrieve_chunks(query, embedded_chunks, top_k=10)
    reranked = rerank_chunks(query, retrieved, top_k=3)
    answer = generate_answer(query, reranked)
    return answer, retrieved, reranked


def format_sources(chunks: list[dict]) -> str:
    """
    整理 source 展示
    """
    if not chunks:
        return "- 无"

    lines = []
    for item in chunks:
        lines.append(
            f"- source: {item.get('source')} | chunk_id: {item.get('chunk_id')} | score: {item.get('score', 'N/A')}"
        )
    return "\n".join(lines)


def build_report(embedded_chunks: list[dict]) -> str:
    """
    生成 markdown 评测报告
    """
    lines = []
    lines.append("# RAG 效果对比实验报告")
    lines.append("")
    lines.append("## 对比方案")
    lines.append("")
    lines.append("1. 不做 RAG，直接问千问")
    lines.append("2. embedding 检索 + 生成")
    lines.append("3. embedding 检索 + rerank + 生成")
    lines.append("")
    lines.append("## 评测维度")
    lines.append("")
    lines.append("- 是否答到点上")
    lines.append("- 是否引用了正确资料")
    lines.append("- 是否胡编")
    lines.append("")

    for idx, question in enumerate(EVAL_QUESTIONS, start=1):
        print(f"正在评测第 {idx} 题：{question}")

        direct_answer = ask_directly(question)
        rag_answer, rag_chunks = rag_without_rerank(question, embedded_chunks)
        rerank_answer, retrieved_chunks, reranked_chunks = rag_with_rerank(question, embedded_chunks)

        lines.append(f"## 问题 {idx}")
        lines.append("")
        lines.append(f"**问题**：{question}")
        lines.append("")

        lines.append("### 方案 1：直接问千问")
        lines.append("")
        lines.append(direct_answer)
        lines.append("")
        lines.append("人工评估：")
        lines.append("- 是否答到点上：")
        lines.append("- 是否引用了正确资料：不适用 / 无引用")
        lines.append("- 是否胡编：")
        lines.append("")

        lines.append("### 方案 2：embedding 检索 + 生成")
        lines.append("")
        lines.append("检索到的 chunk：")
        lines.append(format_sources(rag_chunks))
        lines.append("")
        lines.append("回答：")
        lines.append(rag_answer)
        lines.append("")
        lines.append("人工评估：")
        lines.append("- 是否答到点上：")
        lines.append("- 是否引用了正确资料：")
        lines.append("- 是否胡编：")
        lines.append("")

        lines.append("### 方案 3：embedding 检索 + rerank + 生成")
        lines.append("")
        lines.append("Embedding 初召回 top-10（展示 top-3 代表项）：")
        lines.append(format_sources(retrieved_chunks[:3]))
        lines.append("")
        lines.append("Rerank 后 top-3：")
        lines.append(format_sources(reranked_chunks))
        lines.append("")
        lines.append("回答：")
        lines.append(rerank_answer)
        lines.append("")
        lines.append("人工评估：")
        lines.append("- 是否答到点上：")
        lines.append("- 是否引用了正确资料：")
        lines.append("- 是否胡编：")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 总结")
    lines.append("")
    lines.append("### 哪种方案整体更稳？")
    lines.append("")
    lines.append("（请你根据上面的人工评估填写）")
    lines.append("")
    lines.append("### 哪些题目 rerank 帮助明显？")
    lines.append("")
    lines.append("（请你根据上面的人工评估填写）")
    lines.append("")
    lines.append("### 哪些情况下直接问模型也能答对？")
    lines.append("")
    lines.append("（请你根据上面的人工评估填写）")
    lines.append("")
    lines.append("### 哪些问题最容易胡编？")
    lines.append("")
    lines.append("（请你根据上面的人工评估填写）")
    lines.append("")

    return "\n".join(lines)


def save_report(content: str, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    try:
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)
        report = build_report(embedded_chunks)
        save_report(report, OUTPUT_FILE)
        print(f"\n评测报告已保存到：{Path(OUTPUT_FILE).resolve()}")

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
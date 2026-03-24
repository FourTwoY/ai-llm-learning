import os
import json
import math
from pathlib import Path
from openai import OpenAI


EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
CHAT_MODEL = "qwen3-max-2026-01-23"


def get_client() -> OpenAI:
    """
    创建阿里云百炼 OpenAI 兼容客户端（北京地域）
    如果你是国际站/新加坡地域，请改 base_url
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def load_embeddings(file_path: str) -> list[dict]:
    """
    从本地读取 embeddings 文件
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 embeddings 文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")

    if not isinstance(items, list) or not items:
        raise ValueError("embeddings 文件格式不正确，items 为空或不存在。")

    return items


def embed_query(query: str) -> list[float]:
    """
    把 query 转成 embedding
    """
    if not query.strip():
        raise ValueError("query 不能为空。")

    client = get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
        dimensions=EMBEDDING_DIM,
        encoding_format="float"
    )

    return response.data[0].embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    计算余弦相似度
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def retrieve_top_k(query: str, embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    输入 query，返回最相似的 top-k chunk
    """
    query_embedding = embed_query(query)

    scored_items = []
    for item in embedded_chunks:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_items.append({
            "chunk_id": item["chunk_id"],
            "doc_id": item.get("doc_id"),
            "source": item.get("source"),
            "text": item["text"],
            "score": score
        })

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:top_k]


def build_context(retrieved_chunks: list[dict]) -> str:
    """
    把检索结果拼成上下文
    """
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        part = f"""[参考材料 {i}]
chunk_id: {item["chunk_id"]}
source: {item.get("source")}
text:
{item["text"]}
"""
        context_parts.append(part)

    return "\n\n".join(context_parts)


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    把问题 + 检索上下文一起送给千问生成答案
    """
    client = get_client()
    context = build_context(retrieved_chunks)

    system_prompt = """
你是一个严谨的知识库问答助手。
请严格依据给定的参考材料回答问题。
如果参考材料不足以回答问题，请明确说明“根据当前检索到的材料，无法确定”。
不要编造参考材料中没有出现的事实。
回答尽量清晰、简洁。
""".strip()

    user_prompt = f"""
用户问题：
{query}

参考材料如下：
{context}

请完成以下任务：
1. 先根据参考材料回答用户问题
2. 如果材料不足，请明确说明
3. 最后单独列出你使用到的参考来源 source
""".strip()

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content


def print_retrieved_chunks(retrieved_chunks: list[dict]):
    """
    打印检索结果
    """
    print("\n=== 检索到的 Top-k Chunk ===")
    for i, item in enumerate(retrieved_chunks, start=1):
        print(f"\n【第 {i} 条】")
        print(f"score   : {item['score']:.4f}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"doc_id  : {item.get('doc_id')}")
        print(f"source  : {item.get('source')}")
        print(f"text    : {item['text'][:300]}...")


def main():
    """
    Day 35:
    1. 输入 query
    2. 检索 top-k chunks
    3. 把 chunks 拼成上下文
    4. 送给千问生成答案
    """
    try:
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        query = input("请输入你的问题：").strip()
        if not query:
            print("输入不能为空。")
            return

        retrieved_chunks = retrieve_top_k(query, embedded_chunks, top_k=3)
        print_retrieved_chunks(retrieved_chunks)

        print("\n=== 开始生成答案 ===")
        answer = generate_answer(query, retrieved_chunks)

        print("\n=== RAG 回答结果 ===")
        print(answer)

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
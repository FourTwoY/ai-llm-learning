import os
import json
import math
from pathlib import Path
from openai import OpenAI


CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
BATCH_SIZE = 10


def get_client() -> OpenAI:
    """
    创建阿里云百炼 OpenAI 兼容客户端（北京地域）
    如果你使用的是国际站/新加坡地域，需要改 base_url
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return client


def load_chunks(file_path: str) -> list[dict]:
    """
    从 chunks.json 中读取 chunk
    兼容常见结构，并保留 chunk_id / doc_id / source / text
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 chunks 文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("chunks.json 顶层必须是列表。")

    normalized = []

    for i, item in enumerate(data):
        if isinstance(item, str):
            normalized.append({
                "chunk_id": f"chunk_{i}",
                "doc_id": None,
                "source": None,
                "text": item
            })
        elif isinstance(item, dict):
            text = item.get("text") or item.get("chunk") or item.get("content")
            if not text:
                raise ValueError(f"第 {i} 个 chunk 找不到 text/content/chunk 字段。")

            normalized.append({
                "chunk_id": item.get("chunk_id", f"chunk_{i}"),
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "text": text
            })
        else:
            raise ValueError(f"第 {i} 个 chunk 格式不支持：{type(item)}")

    if not normalized:
        raise ValueError("没有可用于 embedding 的 chunk。")

    return normalized


def embed_texts(texts: list[str], dimensions: int = EMBEDDING_DIM):
    """
    批量生成 embedding
    """
    client = get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=dimensions,
        encoding_format="float"
    )

    embeddings = [item.embedding for item in response.data]
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "model": response.model
    }

    return embeddings, usage


def build_all_embeddings(chunks: list[dict]) -> tuple[list[dict], dict]:
    """
    对所有 chunk 批量 embedding
    """
    all_items = []
    total_prompt_tokens = 0
    total_tokens = 0
    model_name = EMBEDDING_MODEL

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [item["text"] for item in batch]

        embeddings, usage = embed_texts(texts, dimensions=EMBEDDING_DIM)

        total_prompt_tokens += usage["prompt_tokens"]
        total_tokens += usage["total_tokens"]
        model_name = usage["model"]

        for item, emb in zip(batch, embeddings):
            all_items.append({
                "chunk_id": item["chunk_id"],
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "text": item["text"],
                "embedding": emb
            })

    meta = {
        "count": len(all_items),
        "model": model_name,
        "embedding_dim": len(all_items[0]["embedding"]) if all_items else 0,
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens
    }

    return all_items, meta


def save_embeddings(file_path: str, items: list[dict], meta: dict):
    """
    保存 embedding 结果到本地 JSON
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "count": meta["count"],
        "usage": {
            "prompt_tokens": meta["prompt_tokens"],
            "total_tokens": meta["total_tokens"],
            "model": meta["model"],
            "embedding_dim": meta["embedding_dim"]
        },
        "items": items
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_embeddings(file_path: str) -> list[dict]:
    """
    从本地读取已经保存的 embeddings
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 embeddings 文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")

    if not isinstance(items, list) or not items:
        raise ValueError("embeddings 文件格式不正确，items 为空或不存在。")

    return items


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


def embed_query(query: str) -> list[float]:
    """
    把 query 转成 embedding
    """
    if not query.strip():
        raise ValueError("query 不能为空。")

    embeddings, _ = embed_texts([query], dimensions=EMBEDDING_DIM)
    return embeddings[0]


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


def ensure_embeddings_ready():
    """
    如果本地没有 embeddings 文件，就先批量生成
    """
    embeddings_path = Path(EMBEDDINGS_FILE)

    if embeddings_path.exists():
        print(f"检测到已有 embeddings 文件：{embeddings_path}")
        return

    print("未检测到 embeddings 文件，开始批量生成所有 chunk 的向量...")
    chunks = load_chunks(CHUNKS_FILE)
    items, meta = build_all_embeddings(chunks)
    save_embeddings(EMBEDDINGS_FILE, items, meta)
    print(f"向量生成完成，已保存到：{embeddings_path.resolve()}")


def main():
    """
    Day 34:
    1. 确保所有 chunk 已批量向量化
    2. 输入 query
    3. 本地计算 cosine similarity
    4. 返回 top-3 结果
    """
    try:
        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        query = input("请输入你的问题：").strip()
        if not query:
            print("输入不能为空。")
            return

        results = retrieve_top_k(query, embedded_chunks, top_k=3)

        print("\n=== Top-3 检索结果 ===")
        for i, item in enumerate(results, start=1):
            print(f"\n【第 {i} 条】")
            print(f"score   : {item['score']:.4f}")
            print(f"chunk_id: {item['chunk_id']}")
            print(f"doc_id  : {item.get('doc_id')}")
            print(f"source  : {item.get('source')}")
            print(f"text    : {item['text'][:300]}...")

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
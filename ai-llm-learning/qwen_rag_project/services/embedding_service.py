import os
import json
from pathlib import Path
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
BATCH_SIZE = 10


def get_client() -> OpenAI:
    """
    创建阿里云百炼 OpenAI 兼容客户端（北京地域）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def normalize_chunks(chunks: list[dict]) -> list[dict]:
    """
    统一 chunk 结构，至少保留：
    - chunk_id
    - doc_id
    - source
    - text
    """
    normalized = []

    for i, item in enumerate(chunks):
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

    return normalized


def embed_texts(texts: list[str], dimensions: int = EMBEDDING_DIM) -> tuple[list[list[float]], dict]:
    """
    批量生成文本 embedding
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
        "model": response.model,
        "embedding_dim": len(response.data[0].embedding)
    }

    return embeddings, usage


def build_chunk_embeddings(chunks: list[dict]) -> tuple[list[dict], dict]:
    """
    对所有 chunk 批量向量化
    """
    chunks = normalize_chunks(chunks)

    all_items = []
    total_prompt_tokens = 0
    total_tokens = 0
    model_name = EMBEDDING_MODEL
    embedding_dim = EMBEDDING_DIM

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [item["text"] for item in batch]

        embeddings, usage = embed_texts(texts, dimensions=EMBEDDING_DIM)

        total_prompt_tokens += usage["prompt_tokens"]
        total_tokens += usage["total_tokens"]
        model_name = usage["model"]
        embedding_dim = usage["embedding_dim"]

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
        "embedding_dim": embedding_dim,
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens
    }

    return all_items, meta


def save_embeddings(file_path: str, items: list[dict], meta: dict):
    """
    保存 embedding 结果
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
    读取本地 embeddings 文件
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 embeddings 文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")

    if not isinstance(items, list) or not items:
        raise ValueError("embeddings 文件格式不正确，items 为空或不存在。")

    return items
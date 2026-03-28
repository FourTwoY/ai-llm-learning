import os
import json
from pathlib import Path

from openai import OpenAI

from config import get_config


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def normalize_chunks(chunks: list[dict]) -> list[dict]:
    normalized = []
    for i, item in enumerate(chunks):
        if isinstance(item, str):
            normalized.append(
                {
                    "chunk_id": f"chunk_{i}",
                    "doc_id": None,
                    "source": None,
                    "text": item,
                }
            )
        elif isinstance(item, dict):
            text = item.get("text") or item.get("chunk") or item.get("content")
            if not text:
                raise ValueError(f"第 {i} 个 chunk 找不到 text/content/chunk 字段。")

            normalized.append(
                {
                    "chunk_id": item.get("chunk_id", f"chunk_{i}"),
                    "doc_id": item.get("doc_id"),
                    "source": item.get("source"),
                    "text": text,
                }
            )
        else:
            raise ValueError(f"第 {i} 个 chunk 格式不支持：{type(item)}")

    return normalized


def embed_texts(texts: list[str], dimensions: int | None = None) -> tuple[list[list[float]], dict]:
    cfg = get_config()
    model_name = cfg["models"]["embedding_model"]
    embedding_cfg = cfg["embedding"]

    if dimensions is None:
        dimensions = embedding_cfg["dimension"]

    client = get_client()
    response = client.embeddings.create(
        model=model_name,
        input=texts,
        dimensions=dimensions,
        encoding_format="float",
    )

    embeddings = [item.embedding for item in response.data]
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "model": response.model,
        "embedding_dim": len(response.data[0].embedding),
    }
    return embeddings, usage


def build_chunk_embeddings(chunks: list[dict]) -> tuple[list[dict], dict]:
    cfg = get_config()
    embedding_cfg = cfg["embedding"]
    model_name = cfg["models"]["embedding_model"]
    batch_size = embedding_cfg["batch_size"]
    embedding_dim = embedding_cfg["dimension"]

    chunks = normalize_chunks(chunks)
    all_items = []
    total_prompt_tokens = 0
    total_tokens = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_texts = [item["text"] for item in batch]
        batch_embeddings, usage = embed_texts(batch_texts, dimensions=embedding_dim)

        total_prompt_tokens += usage["prompt_tokens"]
        total_tokens += usage["total_tokens"]

        for item, emb in zip(batch, batch_embeddings):
            all_items.append(
                {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item.get("doc_id"),
                    "source": item.get("source"),
                    "text": item["text"],
                    "embedding": emb,
                }
            )

    meta = {
        "count": len(all_items),
        "model": model_name,
        "embedding_dim": embedding_dim,
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
    }

    return all_items, meta


def save_embeddings(file_path: str, items: list[dict], meta: dict):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "items": items}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_embeddings(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 embedding 文件：{file_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    if isinstance(payload, list):
        return payload

    raise ValueError("embedding 文件格式不合法。")
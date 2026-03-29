import json
import os
from pathlib import Path

from openai import OpenAI

from .exceptions import ConfigError, EmbeddingError, DataEmptyError
from .logger_service import log_step, log_result

EMBEDDING_MODEL = "text-embedding-v4"


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ConfigError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def embed_texts(texts: list[str]) -> tuple[list[list[float]], dict]:
    if not texts:
        raise DataEmptyError("embedding 输入文本为空。")

    with log_step("embedding", text_count=len(texts)):
        try:
            client = get_client()
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=1024,
                encoding_format="float",
            )

            vectors = [item.embedding for item in response.data]
            usage = getattr(response, "usage", None)

            log_result("embedding", result_count=len(vectors), extra={"usage": usage})
            return vectors, {"usage": str(usage) if usage else None}

        except ConfigError:
            raise
        except Exception as e:
            raise EmbeddingError(f"embedding 调用失败：{e}") from e


def build_chunk_embeddings(chunks: list[dict]) -> tuple[list[dict], dict]:
    if not chunks:
        raise DataEmptyError("chunks 为空，无法生成 embeddings。")

    with log_step("build_chunk_embeddings", chunk_count=len(chunks)):
        texts = [item["text"] for item in chunks]
        vectors, meta = embed_texts(texts)

        items = []
        for chunk, vector in zip(chunks, vectors):
            items.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id"),
                "source": chunk.get("source"),
                "text": chunk["text"],
                "embedding": vector,
            })

        log_result("build_chunk_embeddings", result_count=len(items))
        return items, meta


def save_embeddings(filepath: str, items: list[dict], meta: dict | None = None):
    with log_step("save_embeddings", filepath=filepath, item_count=len(items)):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "meta": meta or {},
            "items": items,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        log_result("save_embeddings", result_count=len(items), extra={"filepath": filepath})


def load_embeddings(filepath: str) -> list[dict]:
    with log_step("load_embeddings", filepath=filepath):
        path = Path(filepath)
        if not path.exists():
            raise DataEmptyError(f"embeddings 文件不存在：{filepath}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        items = payload.get("items", [])
        if not items:
            raise DataEmptyError("embeddings 文件为空。")

        log_result("load_embeddings", result_count=len(items))
        return items
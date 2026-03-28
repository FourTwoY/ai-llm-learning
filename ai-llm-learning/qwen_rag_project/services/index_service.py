import json
import re
from pathlib import Path

from config import get_config
from .embedding_service import build_chunk_embeddings, save_embeddings


def read_raw_documents(raw_dir: str | None = None) -> list[dict]:
    """读取 data/raw 下的 txt / md 文件"""
    cfg = get_config()
    raw_dir = raw_dir or cfg["paths"]["raw_dir"]

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到原始文档目录：{raw_dir}")

    files = list(raw_path.glob("*.txt")) + list(raw_path.glob("*.md"))
    if not files:
        raise ValueError(f"{raw_dir} 中没有可读取的 txt 或 md 文件。")

    docs = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        doc_id = file_path.stem.lower().replace(" ", "_")
        docs.append(
            {
                "doc_id": doc_id,
                "source": file_path.name,
                "text": text,
            }
        )

    if not docs:
        raise ValueError("没有可用的非空文档。")

    return docs


def save_json(data, file_path: str):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    cfg = get_config()
    chunk_cfg = cfg["chunking"]

    if chunk_size is None:
        chunk_size = chunk_cfg["chunk_size"]
    if overlap is None:
        overlap = chunk_cfg["overlap"]

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则会导致切分步长异常")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks(docs: list[dict], chunk_size: int | None = None, overlap: int | None = None) -> list[dict]:
    all_chunks = []

    for doc in docs:
        doc_chunks = split_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text in enumerate(doc_chunks, start=1):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk{i}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "text": chunk_text,
                }
            )

    if not all_chunks:
        raise ValueError("切分后没有得到任何 chunk。")

    return all_chunks


def rebuild_index() -> dict:
    cfg = get_config(reload=True)
    paths = cfg["paths"]
    chunk_cfg = cfg["chunking"]

    docs = read_raw_documents(paths["raw_dir"])
    save_json(docs, paths["processed_file"])

    chunks = build_chunks(
        docs,
        chunk_size=chunk_cfg["chunk_size"],
        overlap=chunk_cfg["overlap"],
    )
    save_json(chunks, paths["chunks_file"])

    embedding_items, meta = build_chunk_embeddings(chunks)
    save_embeddings(paths["embeddings_file"], embedding_items, meta)

    return {
        "message": "索引重建成功",
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "embedding_count": len(embedding_items),
        "processed_file": paths["processed_file"],
        "chunks_file": paths["chunks_file"],
        "embeddings_file": paths["embeddings_file"],
    }
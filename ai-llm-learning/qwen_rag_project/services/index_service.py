import json
import re
from pathlib import Path

from .embedding_service import build_chunk_embeddings, save_embeddings


RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/docs.json"
CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"


def read_raw_documents(raw_dir: str = RAW_DIR) -> list[dict]:
    """
    读取 data/raw 下的 txt / md 文件
    """
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
        docs.append({
            "doc_id": doc_id,
            "source": file_path.name,
            "text": text
        })

    if not docs:
        raise ValueError("没有可用的非空文档。")

    return docs


def save_json(data, file_path: str):
    """
    通用 JSON 保存函数
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    一个简单可用的文本切分函数
    按字符长度切分，并保留 overlap
    """
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


def build_chunks(docs: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    把文档切成 chunk
    """
    all_chunks = []

    for doc in docs:
        doc_chunks = split_text(doc["text"], chunk_size=chunk_size, overlap=overlap)

        for i, chunk_text in enumerate(doc_chunks, start=1):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk{i}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": chunk_text
            })

    if not all_chunks:
        raise ValueError("切分后没有得到任何 chunk。")

    return all_chunks


def rebuild_index() -> dict:
    """
    重建本地索引：
    1. 重新读取 data/raw
    2. 保存 docs.json
    3. 保存 chunks.json
    4. 生成 embeddings
    """
    docs = read_raw_documents(RAW_DIR)
    save_json(docs, PROCESSED_FILE)

    chunks = build_chunks(docs, chunk_size=500, overlap=100)
    save_json(chunks, CHUNKS_FILE)

    embedding_items, meta = build_chunk_embeddings(chunks)
    save_embeddings(EMBEDDINGS_FILE, embedding_items, meta)

    return {
        "message": "索引重建成功",
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "embedding_count": len(embedding_items),
        "processed_file": PROCESSED_FILE,
        "chunks_file": CHUNKS_FILE,
        "embeddings_file": EMBEDDINGS_FILE
    }
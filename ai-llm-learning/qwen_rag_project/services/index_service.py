from pathlib import Path

from .document_service import read_raw_documents, save_processed_documents, save_chunks
from .embedding_service import build_chunk_embeddings, save_embeddings
from .exceptions import DataEmptyError, IndexBuildError
from .logger_service import log_step, log_result

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/docs.json"
CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if not text.strip():
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

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(docs: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    all_chunks = []

    for doc in docs:
        doc_id = doc["doc_id"]
        source = doc.get("source")
        text = doc["text"]

        pieces = split_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "doc_id": doc_id,
                "source": source,
                "text": piece,
            })

    return all_chunks


def rebuild_index():
    with log_step("rebuild_index", raw_dir=RAW_DIR):
        try:
            docs = read_raw_documents(RAW_DIR)
            if not docs:
                raise DataEmptyError("data/raw 中没有可用文档。")

            save_processed_documents(PROCESSED_FILE, docs)

            chunks = build_chunks(docs, chunk_size=500, overlap=100)
            if not chunks:
                raise DataEmptyError("文档切分后没有得到任何 chunk。")

            save_chunks(CHUNKS_FILE, chunks)

            embedded_items, meta = build_chunk_embeddings(chunks)
            if not embedded_items:
                raise DataEmptyError("没有生成任何 embedding。")

            save_embeddings(EMBEDDINGS_FILE, embedded_items, meta)

            result = {
                "message": "索引重建完成",
                "doc_count": len(docs),
                "chunk_count": len(chunks),
                "embedding_count": len(embedded_items),
                "processed_file": PROCESSED_FILE,
                "chunks_file": CHUNKS_FILE,
                "embeddings_file": EMBEDDINGS_FILE,
            }

            log_result(
                "rebuild_index",
                result_count=len(embedded_items),
                extra={
                    "doc_count": len(docs),
                    "chunk_count": len(chunks),
                    "embedding_count": len(embedded_items),
                }
            )
            return result

        except Exception as e:
            if isinstance(e, DataEmptyError):
                raise
            raise IndexBuildError(f"索引重建失败：{e}") from e
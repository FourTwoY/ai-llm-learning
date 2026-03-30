import json
from pathlib import Path


def _load_json_list(file_path: str, label: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{label} must be a JSON array")

    return data


def read_raw_documents(raw_dir: str) -> list[dict]:
    """
    Read raw `.txt` / `.md` documents from `data/raw`.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")

    files = sorted(
        [p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() in [".txt", ".md"]]
    )

    docs = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        docs.append({
            "doc_id": file_path.stem,
            "source": file_path.name,
            "text": text,
        })

    return docs


def save_processed_documents(file_path: str, docs: list[dict]) -> None:
    """
    Save normalized docs.json.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def save_chunks(file_path: str, chunks: list[dict]) -> None:
    """
    Save chunked chunks.json.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_documents(file_path: str) -> list[dict]:
    """
    Load processed docs.json from local storage.
    """
    return _load_json_list(file_path, "docs")


def load_chunks(file_path: str) -> list[dict]:
    """
    Load chunked chunks.json from local storage.
    """
    return _load_json_list(file_path, "chunks")


def chunk_documents(file_path: str) -> list[dict]:
    """
    Backward-compatible alias for loading chunks.json.
    """
    return load_chunks(file_path)

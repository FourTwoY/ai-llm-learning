from __future__ import annotations

from pathlib import Path

from lora_domain_learning_assistant.src.utils.io import list_supported_files, read_jsonl, read_text_file


def load_raw_documents(raw_dir: str | Path) -> list[dict]:
    docs = []
    for file_path in list_supported_files(raw_dir):
        text = read_text_file(file_path).strip()
        if not text:
            continue
        docs.append(
            {
                "doc_id": file_path.stem,
                "source": file_path.name,
                "path": str(file_path.resolve()),
                "suffix": file_path.suffix.lower(),
                "text": text,
                "char_count": len(text),
                "line_count": len(text.splitlines()),
            }
        )
    return docs


def load_sft_records(jsonl_file: str | Path) -> list[dict]:
    records = []
    for item in read_jsonl(jsonl_file):
        records.append(
            {
                "instruction": str(item.get("instruction", "")).strip(),
                "input": str(item.get("input", "")).strip(),
                "output": str(item.get("output", "")).strip(),
            }
        )
    return records

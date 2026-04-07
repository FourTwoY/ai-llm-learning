from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def write_text_file(path: str | Path, content: str) -> Path:
    target = ensure_parent(path)
    target.write_text(content, encoding="utf-8")
    return target


def read_jsonl(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, records: Iterable[dict]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def list_supported_files(raw_dir: str | Path, suffixes: tuple[str, ...] = (".md", ".txt", ".json")) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)

#!/usr/bin/env python3
"""Day 31: 读取并清洗 data/raw/ 下的 .txt / .md 文档，输出统一结构的 docs.json。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def read_text_file(path: Path) -> str:
    """尽量稳妥地读取 UTF-8 / UTF-8-SIG / GB 系编码文本。"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_text(text: str) -> str:
    """做轻量清洗：统一换行、去掉零宽字符、压缩空白、保留正文信息。"""
    text = normalize_newlines(text)
    # \u00a0：不换行空格，替换成普通空格
    # \u200b：零宽空格，直接删掉    这些字符常出现在网页复制、PDF 转文本、富文本导出中，会影响切分和检索。
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # \ufeff 是 BOM 标记，有时会混进文本开头。删掉它，避免标题提取或正文处理出问题。
    text = text.replace("\ufeff", "")
    # 把连续空格或 tab 压成一个空格
    text = re.sub(r"[ \t]+", " ", text)
    # 如果某一行开头有多余空格或 tab，把它们去掉。
    text = re.sub(r"\n[ \t]+", "\n", text)
    # 如果连续出现 3 个或更多换行，就压缩成 2 个换行。
    '''
    这样做是为了,保留段落之间的空行;但避免空行太多导致文本“松散”甚至影响 chunk
    '''
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(path: Path, text: str) -> str:
    """优先取 Markdown 一级标题；否则取第一行非空内容；再否则回退到文件名。"""
    if path.suffix.lower() == ".md":
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:120]

    '''path.stem = "my_python_notes"
    替换后："my python notes"
    return 结果：my python notes'''
    return path.stem.replace("_", " ").replace("-", " ")


# 保证：同一文件永远生成相同 ID、不同文件 ID 不同、ID 合法可用
def build_doc_id(path: Path, raw_root: Path) -> str:
    # 拿到相对路径
    rel = str(path.relative_to(raw_root)).replace("\\", "/")
    # 根据相对路径得到哈希编码
    digest = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()
    return f"{stem}_{digest}" if stem else f"doc_{digest}"


def load_single_doc(path: Path, raw_root: Path) -> dict[str, Any]:
    raw_text = read_text_file(path)
    text = clean_text(raw_text)
    title = extract_title(path, text)
    return {
        "doc_id": build_doc_id(path, raw_root),
        "source": str(path.relative_to(raw_root)).replace("\\", "/"),
        "title": title,
        "text": text,
    }


def collect_docs(raw_root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(load_single_doc(path, raw_root))
    return docs


# 将 Python 字典列表（文档数据）保存为格式化、可读的 UTF-8 JSON 文件的标准函数
def save_docs(docs: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(docs: list[dict[str, Any]], raw_root: Path, output_path: Path) -> None:
    print(f"原始目录: {raw_root}")
    print(f"输出文件: {output_path}")
    print(f"共处理文档: {len(docs)}")
    if docs:
        print("前 3 个文档示例:")
        for doc in docs[:3]:
            print(f"- {doc['source']} -> {doc['doc_id']} | {doc['title']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and clean docs from data/raw/ into docs.json")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="原始文档目录，默认 data/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/docs.json"),
        help="输出 JSON 路径，默认 data/processed/docs.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_dir
    output_path = args.output

    if not raw_root.exists():
        raise FileNotFoundError(f"找不到原始目录: {raw_root}")

    docs = collect_docs(raw_root)
    save_docs(docs, output_path)
    print_summary(docs, raw_root, output_path)


if __name__ == "__main__":
    main()

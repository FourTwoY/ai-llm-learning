import os
import json
import argparse
from typing import List, Dict


# =============================
# Day 32: 简单纯 Python 文本切分器
# 功能：
# 1. 读取 data/processed/docs.json 中的文档列表
# 2. 对每篇文档做滑动窗口切分
# 3. 输出到 data/chunks/chunks.json
# 4. 在控制台打印统计信息
#
# 约定输入文档格式：
# [
#   {
#     "doc_id": "...",
#     "source": "...",
#     "title": "...",
#     "text": "..."
#   }
# ]
#
# 输出 chunk 格式：
# [
#   {
#     "chunk_id": "doc1_chunk1",
#     "doc_id": "doc1",
#     "text": "...",
#     "source": "xxx.md"
#   }
# ]
# =============================


def ensure_dir(path: str) -> None:
    """如果目录不存在，就递归创建。"""
    os.makedirs(path, exist_ok=True)



def load_docs_from_json(input_path: str) -> List[Dict]:
    """
    从 JSON 文件中读取文档列表。
    如果文件不存在或内容不合法，抛出明确错误，便于排查。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 格式错误：顶层必须是 list")

    return data



def normalize_text(text: str) -> str:
    """
    对文本做轻量清洗，避免切块时出现过多噪声。
    注意：这里不做激进清洗，尽量保留原文信息。
    """
    if not text:
        return ""

    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 去除零宽字符、BOM 等常见不可见字符
    invisible_chars = ["\ufeff", "\u200b", "\u200c", "\u200d"]
    for ch in invisible_chars:
        text = text.replace(ch, "")

    # 去掉每行首尾多余空白，但保留段落结构
    lines = [line.strip() for line in text.split("\n")]

    # 删除连续空行带来的噪声：保留单个空行即可
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        is_empty = (line == "")
        if is_empty:
            if not prev_empty:
                cleaned_lines.append("")
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    text = "\n".join(cleaned_lines).strip()
    return text



def sliding_window_chunk(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    使用最简单的滑动窗口做字符级切分。

    参数说明：
    - chunk_size: 每个 chunk 的目标长度，建议 300~500 之间
    - overlap: 相邻 chunk 的重叠长度，建议 50~100 之间

    实现逻辑：
    - 每次取 text[start:end]
    - 下一次从 start + (chunk_size - overlap) 开始

    说明：
    - 这里按“字符数”切，不按 token 切，足够完成 Day32 入门任务
    - 对中文文档，这种方式通常比按空格切更直接
    """
    text = normalize_text(text)
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则窗口无法前进")

    chunks = []
    step = chunk_size - overlap
    text_length = len(text)
    start = 0

    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_length:
            break
        start += step

    return chunks



def build_chunks(docs: List[Dict], chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    """
    对所有文档执行切分，组装为统一的 chunk 字典列表。
    """
    all_chunks = []

    for doc in docs:
        doc_id = str(doc.get("doc_id", "unknown_doc"))
        source = str(doc.get("source", "unknown_source"))
        text = doc.get("text", "")

        doc_chunks = sliding_window_chunk(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for idx, chunk_text in enumerate(doc_chunks, start=1):
            chunk = {
                "chunk_id": f"{doc_id}_chunk{idx}",
                "doc_id": doc_id,
                "text": chunk_text,
                "source": source,
            }
            all_chunks.append(chunk)

    return all_chunks



def save_chunks(chunks: List[Dict], output_path: str) -> None:
    """将 chunk 列表保存为 JSON 文件。"""
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)



def print_stats(docs: List[Dict], chunks: List[Dict]) -> None:
    """打印统计信息：总文档数、总 chunk 数、平均 chunk 长度。"""
    total_docs = len(docs)
    total_chunks = len(chunks)

    if total_chunks == 0:
        avg_chunk_length = 0.0
    else:
        avg_chunk_length = sum(len(chunk["text"]) for chunk in chunks) / total_chunks

    print("=" * 50)
    print("切分完成")
    print(f"总文档数: {total_docs}")
    print(f"总 chunk 数: {total_chunks}")
    print(f"平均 chunk 长度: {avg_chunk_length:.2f} 字")
    print("=" * 50)



def parse_args() -> argparse.Namespace:
    """命令行参数。默认路径与你前两天的目录结构保持一致。"""
    parser = argparse.ArgumentParser(description="Day32 简单文本切分器")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/docs.json",
        help="输入文档 JSON 路径，默认 data/processed/docs.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chunks/chunks.json",
        help="输出 chunk JSON 路径，默认 data/chunks/chunks.json",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="每个 chunk 的目标长度，建议 300~500，默认 400",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        help="相邻 chunk 的重叠长度，建议 50~100，默认 80",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    docs = load_docs_from_json(args.input)
    chunks = build_chunks(
        docs=docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    save_chunks(chunks, args.output)
    print_stats(docs, chunks)


if __name__ == "__main__":
    main()

import json
from pathlib import Path


# 负责读取已经落盘的文档/切块 JSON 数据，给上层接口直接使用
def load_documents(file_path: str) -> list[dict]:
    """
    从本地读取已经清洗好的文档数据，例如 data/processed/docs.json
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文档文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("docs.json 顶层必须是列表。")

    return data


def chunk_documents(file_path: str) -> list[dict]:
    """
    直接读取已经切分好的 chunks.json
    这里先不重复做切分算法，先把“读取 chunk 结果”封装成服务函数。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 chunk 文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("chunks.json 顶层必须是列表。")

    return data
import os
import json
from pathlib import Path
from openai import OpenAI


def load_chunks(file_path: str, limit: int = 5) -> list[dict]:
    """
    从 chunks.json 中读取 chunk，并统一成：
    [{"chunk_id": "...", "text": "..."}]
    支持几种常见结构：
    1. ["text1", "text2", ...]
    2. [{"text": "..."}, {"text": "..."}]
    3. [{"chunk_id": "...", "text": "..."}]
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 chunks 文件：{file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("chunks.json 顶层必须是列表。")

    normalized = []

    for i, item in enumerate(data[:limit]):
        if isinstance(item, str):
            normalized.append({
                "chunk_id": f"chunk_{i}",
                "doc_id": None,
                "source": None,
                "text": item
            })
        elif isinstance(item, dict):
            text = item.get("text") or item.get("chunk") or item.get("content")
            if not text:
                raise ValueError(f"第 {i} 个 chunk 找不到 text/content/chunk 字段。")

            normalized.append({
                "chunk_id": item.get("chunk_id", f"chunk_{i}"),
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "text": text
            })
        else:
            raise ValueError(f"第 {i} 个 chunk 格式不支持：{type(item)}")

    if not normalized:
        raise ValueError("没有可用于 embedding 的 chunk。")

    return normalized


def get_client() -> OpenAI:
    """
    创建阿里云百炼 OpenAI 兼容客户端（北京地域）
    如果你用的是新加坡地域，把 base_url 改成：
    https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return client


def embed_chunks(chunks: list[dict], dimensions: int = 1024) -> tuple[list[dict], dict]:
    """
    对多个 chunk 批量生成 embedding
    返回：
    - embeddings_result: [{chunk_id, text, embedding}]
    - usage_info: token 使用信息
    """
    client = get_client()

    texts = [item["text"] for item in chunks]

    response = client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
        dimensions=dimensions,
        encoding_format="float"
    )

    result = []
    for i, item in enumerate(chunks):
        result.append({
            "chunk_id": item["chunk_id"],
            "doc_id": item.get("doc_id"),
            "source": item.get("source"),
            "text": item["text"],
            "embedding": response.data[i].embedding
        })

    usage_info = {
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "model": response.model,
        "embedding_dim": len(response.data[0].embedding)
    }

    return result, usage_info


def save_embeddings(output_path: str, embeddings_result: list[dict], usage_info: dict):
    """
    保存 embedding 结果到本地 JSON
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "count": len(embeddings_result),
        "usage": usage_info,
        "items": embeddings_result
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main():
    """
    Day 33:
    1. 读取 chunks
    2. 取前 3~5 个 chunk
    3. 调 embedding
    4. 保存结果到本地
    """
    try:
        input_file = "data/chunks/chunks.json"
        output_file = "data/embeddings/sample_embeddings.json"

        print("开始读取 chunks...")
        chunks = load_chunks(input_file, limit=5)
        print(f"成功读取 {len(chunks)} 个 chunk。")

        print("开始调用 embedding 接口...")
        embeddings_result, usage_info = embed_chunks(chunks, dimensions=1024)

        print("向量生成成功。")
        print(f"模型名称：{usage_info['model']}")
        print(f"向量维度：{usage_info['embedding_dim']}")
        print(f"prompt_tokens：{usage_info['prompt_tokens']}")
        print(f"total_tokens：{usage_info['total_tokens']}")

        save_embeddings(output_file, embeddings_result, usage_info)
        print(f"结果已保存到：{Path(output_file).resolve()}")

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
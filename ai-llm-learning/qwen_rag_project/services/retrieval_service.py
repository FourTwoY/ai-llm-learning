import math
from .embedding_service import embed_texts


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    计算余弦相似度
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def retrieve_chunks(query: str, embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    输入 query，返回最相似的 top-k chunk
    """
    if not query.strip():
        raise ValueError("query 不能为空。")

    query_embedding, _ = embed_texts([query])
    query_vector = query_embedding[0]

    scored_items = []
    for item in embedded_chunks:
        score = cosine_similarity(query_vector, item["embedding"])
        scored_items.append({
            "chunk_id": item["chunk_id"],
            "doc_id": item.get("doc_id"),
            "source": item.get("source"),
            "text": item["text"],
            "score": score
        })

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:top_k]

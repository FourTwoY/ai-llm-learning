import re
from collections import Counter

from .embedding_service import embed_texts
from .retrieval_service import cosine_similarity


def tokenize(text: str) -> list[str]:
    """
    一个非常简化的 tokenizer：
    - 英文 / 数字 / 下划线 / 连字符 / 点号 作为一个 token
    - 连续中文作为一个 token
    """
    if not text:
        return []

    text = text.lower()
    tokens = re.findall(r"[a-z0-9_+\-\.]+|[\u4e00-\u9fff]+", text)
    return [t.strip() for t in tokens if t.strip()]


def keyword_match_score(query: str, chunk_text: str, source: str | None = None) -> float:
    """
    简化关键词匹配分：
    1. query token 命中数
    2. query token 命中比例
    3. source（文件名）命中加分
    4. query 整体短语子串命中加分

    最终分数压到 0~1
    """
    query_tokens = tokenize(query)
    text_tokens = tokenize(chunk_text)
    source_tokens = tokenize(source or "")

    if not query_tokens or not text_tokens:
        return 0.0

    text_counter = Counter(text_tokens)
    query_unique = list(dict.fromkeys(query_tokens))

    hit_count = 0
    total_freq = 0

    for token in query_unique:
        if token in text_counter:
            hit_count += 1
            total_freq += text_counter[token]

    hit_ratio = hit_count / max(len(query_unique), 1)

    # 频次分做一个轻量裁剪
    freq_score = min(total_freq / max(len(query_unique), 1), 2.0) / 2.0

    # 文件名命中 bonus（近似“标题命中加分”）
    source_hit = 0
    for token in query_unique:
        if token in source_tokens:
            source_hit += 1
    source_bonus = min(source_hit * 0.15, 0.3)

    # query 整体短语作为子串出现时，加一点 bonus
    normalized_query = " ".join(query_tokens)
    normalized_text = " ".join(text_tokens)
    phrase_bonus = 0.15 if normalized_query and normalized_query in normalized_text else 0.0

    raw_score = (
        0.55 * hit_ratio +
        0.20 * freq_score +
        0.10 * source_bonus +
        0.15 * phrase_bonus
    )

    return max(0.0, min(raw_score, 1.0))


def normalize_vector_score(score: float) -> float:
    """
    余弦相似度通常在 [-1, 1]，映射到 [0, 1]
    """
    normalized = (score + 1.0) / 2.0
    return max(0.0, min(normalized, 1.0))


def hybrid_retrieve_chunks(
    query: str,
    embedded_chunks: list[dict],
    top_k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """
    hybrid retrieval v1:
    - 保留 embedding 检索
    - 增加关键词匹配分
    - 两者加权融合
    """
    if not query.strip():
        raise ValueError("query 不能为空。")

    if not embedded_chunks:
        raise ValueError("embedded_chunks 不能为空。")

    if abs((vector_weight + keyword_weight) - 1.0) > 1e-8:
        raise ValueError("vector_weight 和 keyword_weight 之和必须等于 1.0。")

    query_embeddings, _ = embed_texts([query])
    query_vector = query_embeddings[0]

    scored_items = []

    for item in embedded_chunks:
        vector_score_raw = cosine_similarity(query_vector, item["embedding"])
        vector_score = normalize_vector_score(vector_score_raw)

        keyword_score = keyword_match_score(
            query=query,
            chunk_text=item["text"],
            source=item.get("source"),
        )

        hybrid_score = vector_weight * vector_score + keyword_weight * keyword_score

        scored_items.append({
            "chunk_id": item["chunk_id"],
            "doc_id": item.get("doc_id"),
            "source": item.get("source"),
            "text": item["text"],
            "vector_score_raw": vector_score_raw,
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "score": hybrid_score,
        })

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:top_k]
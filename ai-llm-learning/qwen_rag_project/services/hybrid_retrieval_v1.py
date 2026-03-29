import re
from collections import Counter

from .embedding_service import embed_texts
from .retrieval_service import cosine_similarity
from .exceptions import InvalidRequestError, DataEmptyError
from .logger_service import log_step, log_result


def tokenize(text: str) -> list[str]:
    if not text:
        return []

    text = text.lower()
    tokens = re.findall(r"[a-z0-9_+\-\.]+|[\u4e00-\u9fff]+", text)
    return [t.strip() for t in tokens if t.strip()]


def keyword_match_score(query: str, chunk_text: str, source: str | None = None) -> float:
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
    freq_score = min(total_freq / max(len(query_unique), 1), 2.0) / 2.0

    source_hit = 0
    for token in query_unique:
        if token in source_tokens:
            source_hit += 1
    source_bonus = min(source_hit * 0.15, 0.3)

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
    normalized = (score + 1.0) / 2.0
    return max(0.0, min(normalized, 1.0))


def hybrid_retrieve_chunks(
    query: str,
    embedded_chunks: list[dict],
    top_k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    if not query.strip():
        raise InvalidRequestError("query 不能为空。")
    if not embedded_chunks:
        raise DataEmptyError("embedded_chunks 为空。")
    if abs((vector_weight + keyword_weight) - 1.0) > 1e-8:
        raise InvalidRequestError("vector_weight 和 keyword_weight 之和必须等于 1.0。")

    with log_step(
        "hybrid_retrieval",
        query=query,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        candidate_count=len(embedded_chunks),
    ):
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
        results = scored_items[:top_k]

        log_result("hybrid_retrieval", result_count=len(results))
        return results
from http import HTTPStatus

import dashscope

from config import get_config


def rerank_chunks(query: str, retrieved_chunks: list[dict], top_k: int | None = None) -> list[dict]:
    cfg = get_config()
    model_name = cfg["models"]["rerank_model"]
    retrieval_cfg = cfg["retrieval"]

    if top_k is None:
        top_k = retrieval_cfg["rerank_top_n"]

    if not query.strip():
        raise ValueError("query 不能为空。")
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks 不能为空。")

    documents = [item["text"] for item in retrieved_chunks]
    resp = dashscope.TextReRank.call(
        model=model_name,
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
        return_documents=True,
        instruct="Given a user question, retrieve the most relevant passages that answer the question.",
    )

    if resp.status_code != HTTPStatus.OK:
        raise ValueError(f"rerank 调用失败：{resp}")

    results = resp.output["results"]
    reranked = []
    for item in results:
        original_index = item["index"]
        original_chunk = retrieved_chunks[original_index]
        reranked.append(
            {
                "chunk_id": original_chunk["chunk_id"],
                "doc_id": original_chunk.get("doc_id"),
                "source": original_chunk.get("source"),
                "text": original_chunk["text"],
                "score": item["relevance_score"],
            }
        )
    return reranked
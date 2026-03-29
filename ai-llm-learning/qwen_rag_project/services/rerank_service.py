import dashscope
from http import HTTPStatus

from .exceptions import InvalidRequestError, DataEmptyError, RerankError
from .logger_service import log_step, log_result

RERANK_MODEL = "qwen3-rerank"


def rerank_chunks(query: str, retrieved_chunks: list[dict], top_k: int = 3) -> list[dict]:
    if not query.strip():
        raise InvalidRequestError("query 不能为空。")
    if not retrieved_chunks:
        raise DataEmptyError("retrieved_chunks 不能为空。")

    with log_step("rerank", query=query, input_count=len(retrieved_chunks), top_k=top_k):
        documents = [item["text"] for item in retrieved_chunks]

        try:
            resp = dashscope.TextReRank.call(
                model=RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=True,
                instruct="Given a user question, retrieve the most relevant passages that answer the question."
            )

            if resp.status_code != HTTPStatus.OK:
                raise RerankError(f"rerank 返回异常：status_code={resp.status_code}, body={resp}")

            results = resp.output["results"]
            reranked = []

            for item in results:
                original_index = item["index"]
                original_chunk = retrieved_chunks[original_index]
                reranked.append({
                    "chunk_id": original_chunk["chunk_id"],
                    "doc_id": original_chunk.get("doc_id"),
                    "source": original_chunk.get("source"),
                    "text": original_chunk["text"],
                    "score": item["relevance_score"],
                    "vector_score": original_chunk.get("vector_score"),
                    "keyword_score": original_chunk.get("keyword_score"),
                })

            log_result("rerank", result_count=len(reranked))
            return reranked

        except RerankError:
            raise
        except Exception as e:
            raise RerankError(f"rerank 调用失败：{e}") from e
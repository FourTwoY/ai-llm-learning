from pathlib import Path
from fastapi import FastAPI, HTTPException

from schemas import (
    AskRequest,
    AskResponse,
    ReferenceItem,
    RebuildIndexResponse,
    SearchRequest,
    SearchResultItem,
    SearchResponse,
    ErrorResponse,
)
from services.document_service import chunk_documents
from services.embedding_service import (
    build_chunk_embeddings,
    save_embeddings,
    load_embeddings,
)
from services.retrieval_service import retrieve_chunks
from services.hybrid_retrieval_v1 import hybrid_retrieve_chunks
from services.rerank_service import rerank_chunks
from services.generation_service import generate_answer
from services.index_service import rebuild_index
from services.query_rewrite_service import rewrite_query

CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"

app = FastAPI(
    title="Qwen RAG API",
    version="1.3.0",
    description="一个基于 FastAPI + 阿里云百炼的简易 RAG 问答接口，支持 query rewrite + hybrid retrieval。"
)


def ensure_embeddings_ready():
    """
    如果本地没有 embeddings 文件，就先自动生成
    """
    embeddings_path = Path(EMBEDDINGS_FILE)
    if embeddings_path.exists():
        return

    chunks = chunk_documents(CHUNKS_FILE)
    items, meta = build_chunk_embeddings(chunks)
    save_embeddings(EMBEDDINGS_FILE, items, meta)


@app.get("/")
def read_root():
    return {
        "project_name": "qwen_rag_project",
        "version": "1.3.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入错误"},
        500: {"model": ErrorResponse, "description": "服务内部错误"}
    },
    summary="RAG 问答接口",
    description="输入问题，先做 query rewrite，再做 hybrid retrieval / embedding retrieval，最后生成答案，并返回引用来源。"
)
def ask(request: AskRequest):
    try:
        original_question = request.question.strip()
        if not original_question:
            raise HTTPException(status_code=400, detail="question 不能为空。")

        if abs((request.vector_weight + request.keyword_weight) - 1.0) > 1e-8:
            raise HTTPException(status_code=400, detail="vector_weight 和 keyword_weight 之和必须等于 1.0。")

        rewritten_query = rewrite_query(original_question) if request.use_rewrite else original_question

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        if request.use_hybrid:
            retrieved_chunks = hybrid_retrieve_chunks(
                query=rewritten_query,
                embedded_chunks=embedded_chunks,
                top_k=request.top_k,
                vector_weight=request.vector_weight,
                keyword_weight=request.keyword_weight,
            )
        else:
            retrieved_chunks = retrieve_chunks(
                rewritten_query,
                embedded_chunks,
                top_k=request.top_k
            )

        if request.use_rerank:
            final_chunks = rerank_chunks(
                rewritten_query,
                retrieved_chunks,
                top_k=min(3, len(retrieved_chunks))
            )
        else:
            final_chunks = retrieved_chunks[:min(3, len(retrieved_chunks))]

        answer = generate_answer(original_question, final_chunks)

        references = [
            ReferenceItem(
                source=item.get("source"),
                chunk_id=item["chunk_id"],
                score=float(item["score"])
            )
            for item in final_chunks
        ]

        return AskResponse(
            original_question=original_question,
            rewritten_query=rewritten_query,
            answer=answer,
            references=references
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务内部错误：{e}")


@app.post(
    "/rebuild_index",
    response_model=RebuildIndexResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入或文件错误"},
        500: {"model": ErrorResponse, "description": "索引重建失败"}
    },
    summary="重建知识库索引",
    description="重新读取 data/raw，重新切分文档并生成最新的 embeddings 本地索引。"
)
def rebuild_index_api():
    try:
        result = rebuild_index()
        return RebuildIndexResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引重建失败：{e}")


@app.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入错误"},
        500: {"model": ErrorResponse, "description": "检索失败"}
    },
    summary="检索调试接口",
    description="输入问题，先做 query rewrite，再返回 embedding、hybrid、rerank 后结果。"
)
def search(request: SearchRequest):
    try:
        original_question = request.question.strip()
        if not original_question:
            raise HTTPException(status_code=400, detail="question 不能为空。")

        if abs((request.vector_weight + request.keyword_weight) - 1.0) > 1e-8:
            raise HTTPException(status_code=400, detail="vector_weight 和 keyword_weight 之和必须等于 1.0。")

        rewritten_query = rewrite_query(original_question) if request.use_rewrite else original_question

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        # 1) 纯 embedding 初召回
        embedding_results = retrieve_chunks(
            rewritten_query,
            embedded_chunks,
            top_k=request.top_k
        )

        # 2) hybrid 检索
        if request.use_hybrid:
            hybrid_results = hybrid_retrieve_chunks(
                query=rewritten_query,
                embedded_chunks=embedded_chunks,
                top_k=request.top_k,
                vector_weight=request.vector_weight,
                keyword_weight=request.keyword_weight,
            )
            rerank_input = hybrid_results
        else:
            hybrid_results = []
            rerank_input = embedding_results

        # 3) rerank
        if request.use_rerank:
            rerank_results = rerank_chunks(
                rewritten_query,
                rerank_input,
                top_k=min(3, len(rerank_input))
            )
        else:
            rerank_results = []

        embedding_items = [
            SearchResultItem(
                chunk_id=item["chunk_id"],
                source=item.get("source"),
                score=float(item["score"]),
                text=item["text"],
                vector_score=None,
                keyword_score=None,
            )
            for item in embedding_results
        ]

        hybrid_items = [
            SearchResultItem(
                chunk_id=item["chunk_id"],
                source=item.get("source"),
                score=float(item["score"]),
                text=item["text"],
                vector_score=float(item.get("vector_score", 0.0)),
                keyword_score=float(item.get("keyword_score", 0.0)),
            )
            for item in hybrid_results
        ]

        rerank_items = [
            SearchResultItem(
                chunk_id=item["chunk_id"],
                source=item.get("source"),
                score=float(item["score"]),
                text=item["text"],
                vector_score=float(item.get("vector_score")) if item.get("vector_score") is not None else None,
                keyword_score=float(item.get("keyword_score")) if item.get("keyword_score") is not None else None,
            )
            for item in rerank_results
        ]

        return SearchResponse(
            original_question=original_question,
            rewritten_query=rewritten_query,
            embedding_results=embedding_items,
            hybrid_results=hybrid_items,
            rerank_results=rerank_items
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")
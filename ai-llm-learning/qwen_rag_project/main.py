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
from services.rerank_service import rerank_chunks
from services.generation_service import generate_answer
from services.index_service import rebuild_index


CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"

app = FastAPI(
    title="Qwen RAG API",
    version="1.1.0",
    description="一个基于 FastAPI + 阿里云百炼的简易 RAG 问答接口，支持重建本地知识库索引。"
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
        "version": "1.1.0",
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
    description="输入一个问题，先检索知识库，再生成答案，并返回引用来源。"
)
def ask(request: AskRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空。")

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        if request.use_rerank:
            retrieved_chunks = retrieve_chunks(question, embedded_chunks, top_k=request.top_k)
            final_chunks = rerank_chunks(question, retrieved_chunks, top_k=min(3, len(retrieved_chunks)))
        else:
            final_chunks = retrieve_chunks(question, embedded_chunks, top_k=min(3, request.top_k))

        answer = generate_answer(question, final_chunks)

        references = [
            ReferenceItem(
                source=item.get("source"),
                chunk_id=item["chunk_id"],
                score=float(item["score"])
            )
            for item in final_chunks
        ]

        return AskResponse(
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
    description="输入问题，返回 embedding 初召回结果和 rerank 后结果，方便观察检索效果。"
)
def search(request: SearchRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空。")

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        # 第一步：embedding 初召回
        embedding_results = retrieve_chunks(question, embedded_chunks, top_k=request.top_k)

        # 第二步：可选 rerank
        if request.use_rerank:
            rerank_results = rerank_chunks(
                question,
                embedding_results,
                top_k=min(3, len(embedding_results))
            )
        else:
            rerank_results = []

        embedding_items = [
            SearchResultItem(
                chunk_id=item["chunk_id"],
                source=item.get("source"),
                score=float(item["score"]),
                text=item["text"]
            )
            for item in embedding_results
        ]

        rerank_items = [
            SearchResultItem(
                chunk_id=item["chunk_id"],
                source=item.get("source"),
                score=float(item["score"]),
                text=item["text"]
            )
            for item in rerank_results
        ]

        return SearchResponse(
            embedding_results=embedding_items,
            rerank_results=rerank_items
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")
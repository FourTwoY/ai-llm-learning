from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
from services.logger_service import logger, set_request_id, log_step, log_result
from services.exceptions import AppError, InvalidRequestError

CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.json"

app = FastAPI(
    title="Qwen RAG API",
    version="1.4.0",
    description="一个基于 FastAPI + 阿里云百炼的简易 RAG 问答接口，支持 query rewrite + hybrid retrieval + 日志 + 统一异常处理。"
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = set_request_id()
    logger.info(f"[http_request] START | method={request.method} | path={request.url.path}")
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(f"[http_request] END | method={request.method} | path={request.url.path} | status_code={response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"[http_request] ERROR | method={request.method} | path={request.url.path} | error={e}")
        raise


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    logger.warning(f"[app_error] code={exc.code} | message={exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "error_code": exc.code,
            "path": request.url.path,
        }
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    logger.warning(f"[validation_error] errors={exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数校验失败",
            "error_code": "REQUEST_VALIDATION_ERROR",
            "errors": exc.errors(),
            "path": request.url.path,
        }
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception(f"[unexpected_error] path={request.url.path} | error={exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部异常，请查看日志定位问题。",
            "error_code": "INTERNAL_SERVER_ERROR",
            "path": request.url.path,
        }
    )


def ensure_embeddings_ready():
    embeddings_path = Path(EMBEDDINGS_FILE)
    if embeddings_path.exists():
        return

    with log_step("ensure_embeddings_ready", embeddings_file=EMBEDDINGS_FILE):
        chunks = chunk_documents(CHUNKS_FILE)
        items, meta = build_chunk_embeddings(chunks)
        save_embeddings(EMBEDDINGS_FILE, items, meta)
        log_result("ensure_embeddings_ready", result_count=len(items))


@app.get("/")
def read_root():
    return {
        "project_name": "qwen_rag_project",
        "version": "1.4.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    original_question = request.question.strip()
    if not original_question:
        raise InvalidRequestError("question 不能为空。")

    if abs((request.vector_weight + request.keyword_weight) - 1.0) > 1e-8:
        raise InvalidRequestError("vector_weight 和 keyword_weight 之和必须等于 1.0。")

    with log_step(
        "ask_api",
        question=original_question,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
        use_rewrite=request.use_rewrite,
        use_hybrid=request.use_hybrid,
    ):
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

        log_result("ask_api", result_count=len(references), extra={"rewritten_query": rewritten_query})

        return AskResponse(
            original_question=original_question,
            rewritten_query=rewritten_query,
            answer=answer,
            references=references
        )


@app.post("/rebuild_index", response_model=RebuildIndexResponse)
def rebuild_index_api():
    with log_step("rebuild_index_api"):
        result = rebuild_index()
        log_result("rebuild_index_api", result_count=result["embedding_count"])
        return RebuildIndexResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    original_question = request.question.strip()
    if not original_question:
        raise InvalidRequestError("question 不能为空。")

    if abs((request.vector_weight + request.keyword_weight) - 1.0) > 1e-8:
        raise InvalidRequestError("vector_weight 和 keyword_weight 之和必须等于 1.0。")

    with log_step(
        "search_api",
        question=original_question,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
        use_rewrite=request.use_rewrite,
        use_hybrid=request.use_hybrid,
    ):
        rewritten_query = rewrite_query(original_question) if request.use_rewrite else original_question

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(EMBEDDINGS_FILE)

        embedding_results = retrieve_chunks(
            rewritten_query,
            embedded_chunks,
            top_k=request.top_k
        )

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

        log_result(
            "search_api",
            extra={
                "embedding_count": len(embedding_items),
                "hybrid_count": len(hybrid_items),
                "rerank_count": len(rerank_items),
                "rewritten_query": rewritten_query,
            }
        )

        return SearchResponse(
            original_question=original_question,
            rewritten_query=rewritten_query,
            embedding_results=embedding_items,
            hybrid_results=hybrid_items,
            rerank_results=rerank_items
        )
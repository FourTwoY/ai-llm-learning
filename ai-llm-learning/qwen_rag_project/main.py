from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import (
    AskRequest,
    AskResponse,
    AskData,
    ReferenceItem,
    RebuildIndexResponse,
    RebuildIndexData,
    SearchRequest,
    SearchResultItem,
    SearchResponse,
    SearchData,
    ErrorResponse,
    ErrorData,
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
from services.logger_service import logger, set_request_id, get_request_id, log_step, log_result
from services.exceptions import AppError, InvalidRequestError

from config import get_config
import logging
from config import get_config

cfg = get_config()

logging.basicConfig(
    level=getattr(logging, cfg["logging"]["level"].upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.getLogger(__name__).debug("当前日志级别已设置为 DEBUG（如果你能看到这行，说明 dev 生效）")


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
    return build_error_response(
        message=exc.message,
        error_code=exc.code,
        path=request.url.path,
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    logger.warning(f"[validation_error] errors={exc.errors()}")
    return build_error_response(
        message="请求参数校验失败",
        error_code="REQUEST_VALIDATION_ERROR",
        path=request.url.path,
        status_code=422,
        errors=exc.errors(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception(f"[unexpected_error] path={request.url.path} | error={exc}")
    return build_error_response(
        message="服务器内部异常，请查看日志定位问题。",
        error_code="INTERNAL_SERVER_ERROR",
        path=request.url.path,
        status_code=500,
    )


def ensure_embeddings_ready():
    cfg = get_config()
    chunks_file = cfg["paths"]["chunks_file"]
    embeddings_file = cfg["paths"]["embeddings_file"]

    embeddings_path = Path(embeddings_file)
    if embeddings_path.exists():
        return

    with log_step("ensure_embeddings_ready", embeddings_file=embeddings_file):
        chunks = chunk_documents(chunks_file)
        items, meta = build_chunk_embeddings(chunks)
        save_embeddings(embeddings_file, items, meta)
        log_result("ensure_embeddings_ready", result_count=len(items))

def build_success_response(message: str, data):
    return {
        "success": True,
        "message": message,
        "data": data,
        "trace_id": get_request_id(),
    }


def build_error_response(message: str, error_code: str, path: str, status_code: int, errors: list[dict] | None = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "data": {
                "error_code": error_code,
                "path": path,
                "errors": errors,
            },
            "trace_id": get_request_id(),
        }
    )


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
    cfg = get_config()

    original_question = request.question.strip()
    if not original_question:
        raise InvalidRequestError("question 不能为空。")

    top_k = request.top_k if request.top_k is not None else cfg["retrieval"]["top_k"]
    use_rerank = request.use_rerank if request.use_rerank is not None else True
    use_rewrite = request.use_rewrite if request.use_rewrite is not None else cfg["rewrite"]["use_rewrite"]
    use_hybrid = request.use_hybrid if request.use_hybrid is not None else cfg["retrieval"]["use_hybrid"]
    vector_weight = request.vector_weight if request.vector_weight is not None else cfg["retrieval"]["vector_weight"]
    keyword_weight = request.keyword_weight if request.keyword_weight is not None else cfg["retrieval"]["keyword_weight"]
    rerank_top_n = cfg["retrieval"]["rerank_top_n"]

    if abs((vector_weight + keyword_weight) - 1.0) > 1e-8:
        raise InvalidRequestError("vector_weight 和 keyword_weight 之和必须等于 1.0。")

    with log_step(
        "ask_api",
        question=original_question,
        top_k=top_k,
        use_rerank=use_rerank,
        use_rewrite=use_rewrite,
        use_hybrid=use_hybrid,
    ):
        rewritten_query = rewrite_query(original_question) if use_rewrite else original_question

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(cfg["paths"]["embeddings_file"])

        if use_hybrid:
            retrieved_chunks = hybrid_retrieve_chunks(
                query=rewritten_query,
                embedded_chunks=embedded_chunks,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
        else:
            retrieved_chunks = retrieve_chunks(
                rewritten_query,
                embedded_chunks,
                top_k=top_k
            )

        if use_rerank:
            final_chunks = rerank_chunks(
                rewritten_query,
                retrieved_chunks,
                top_k=min(rerank_top_n, len(retrieved_chunks))
            )
        else:
            final_chunks = retrieved_chunks[:min(rerank_top_n, len(retrieved_chunks))]

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
            success=True,
            message="问答成功",
            data=AskData(
                original_question=original_question,
                rewritten_query=rewritten_query,
                answer=answer,
                references=references,
            ),
            trace_id=get_request_id(),
        )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    cfg = get_config()

    original_question = request.question.strip()
    if not original_question:
        raise InvalidRequestError("question 不能为空。")

    top_k = request.top_k if request.top_k is not None else cfg["retrieval"]["top_k"]
    use_rerank = request.use_rerank if request.use_rerank is not None else True
    use_rewrite = request.use_rewrite if request.use_rewrite is not None else cfg["rewrite"]["use_rewrite"]
    use_hybrid = request.use_hybrid if request.use_hybrid is not None else cfg["retrieval"]["use_hybrid"]
    vector_weight = request.vector_weight if request.vector_weight is not None else cfg["retrieval"]["vector_weight"]
    keyword_weight = request.keyword_weight if request.keyword_weight is not None else cfg["retrieval"]["keyword_weight"]
    rerank_top_n = cfg["retrieval"]["rerank_top_n"]

    if abs((vector_weight + keyword_weight) - 1.0) > 1e-8:
        raise InvalidRequestError("vector_weight 和 keyword_weight 之和必须等于 1.0。")

    with log_step(
        "search_api",
        question=original_question,
        top_k=top_k,
        use_rerank=use_rerank,
        use_rewrite=use_rewrite,
        use_hybrid=use_hybrid,
    ):
        rewritten_query = rewrite_query(original_question) if use_rewrite else original_question

        ensure_embeddings_ready()
        embedded_chunks = load_embeddings(cfg["paths"]["embeddings_file"])

        embedding_results = retrieve_chunks(
            rewritten_query,
            embedded_chunks,
            top_k=top_k
        )

        if use_hybrid:
            hybrid_results = hybrid_retrieve_chunks(
                query=rewritten_query,
                embedded_chunks=embedded_chunks,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
            rerank_input = hybrid_results
        else:
            hybrid_results = []
            rerank_input = embedding_results

        if use_rerank:
            rerank_results = rerank_chunks(
                rewritten_query,
                rerank_input,
                top_k=min(rerank_top_n, len(rerank_input))
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
            success=True,
            message="检索成功",
            data=SearchData(
                original_question=original_question,
                rewritten_query=rewritten_query,
                embedding_results=embedding_items,
                hybrid_results=hybrid_items,
                rerank_results=rerank_items,
            ),
            trace_id=get_request_id(),
        )


@app.post("/rebuild_index", response_model=RebuildIndexResponse)
def rebuild_index_api():
    with log_step("rebuild_index_api"):
        result = rebuild_index()
        log_result("rebuild_index_api", result_count=result["embedding_count"])

        return RebuildIndexResponse(
            success=True,
            message=result["message"],
            data=RebuildIndexData(
                doc_count=result["doc_count"],
                chunk_count=result["chunk_count"],
                embedding_count=result["embedding_count"],
                processed_file=result["processed_file"],
                chunks_file=result["chunks_file"],
                embeddings_file=result["embeddings_file"],
            ),
            trace_id=get_request_id(),
        )
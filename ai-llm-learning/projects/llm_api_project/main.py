from fastapi import FastAPI, HTTPException

from schemas import (
    PaperRequest,
    PaperResponse,
    KeywordRequest,
    KeywordResponse,
    ErrorResponse,
)
from services.llm_service import analyze_paper, extract_keywords

app = FastAPI(
    title="AI LLM Learning API",
    version="0.6.0",
    description="一个基于 FastAPI + 千问 API 的论文摘要分析与关键词提取接口示例。"
)


@app.get("/")
def read_root():
    return {
        "project_name": "llm_api_project",
        "version": "0.6.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"


@app.post(
    "/analyze",
    response_model=PaperResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入错误或返回格式异常"},
        500: {"model": ErrorResponse, "description": "模型调用失败"}
    },
    summary="分析论文摘要",
    description="接收论文摘要文本，调用千问模型后返回结构化分析结果。"
)
def analyze(request: PaperRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="输入 text 不能为空。")

        result = analyze_paper(
            text=request.text,
            style=request.style,
            max_points=request.max_points
        )
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务内部错误：{e}")


@app.post(
    "/keywords",
    response_model=KeywordResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入错误或返回格式异常"},
        500: {"model": ErrorResponse, "description": "模型调用失败"}
    },
    summary="提取关键词",
    description="接收原始文本，调用千问模型后返回关键词列表。"
)
def keywords(request: KeywordRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="输入 text 不能为空。")

        result = extract_keywords(
            text=request.text,
            top_k=request.top_k
        )
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务内部错误：{e}")
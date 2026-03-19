from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from paper_service import analyze_paper

app = FastAPI(
    title="AI LLM Learning API",
    version="0.5.0",
    description="一个基于 FastAPI + 千问 API 的论文摘要分析接口示例。"
)


class PaperRequest(BaseModel):
    text: str = Field(..., description="需要分析的论文摘要文本")
    style: str = Field(default="bullet", description="输出风格，例如 bullet / concise")
    max_points: int = Field(default=5, ge=1, le=10, description="最多输出几点，范围 1~10")


class PaperResponse(BaseModel):
    topic: str = Field(..., description="论文主题")
    research_problem: str = Field(..., description="论文试图解决的核心问题")
    method: str = Field(..., description="论文采用的方法")
    contributions: list[str] = Field(..., description="论文的主要贡献或创新点")
    limitations: list[str] = Field(..., description="论文的主要局限性")
    keywords: list[str] = Field(..., description="论文关键词列表")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误信息说明")


@app.get("/")
def read_root():
    return {
        "project_name": "ai-llm-learning",
        "version": "0.5.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"


@app.post(
    "/analyze",
    response_model=PaperResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入错误"},
        500: {"model": ErrorResponse, "description": "模型调用失败或返回异常"}
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
        # 通常是输入为空、模型返回 JSON 格式异常等
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # 通常是外部 API 调用失败等
        raise HTTPException(status_code=500, detail=f"服务内部错误：{e}")
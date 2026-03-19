from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="AI LLM Learning API",
    version="0.3.0"
)


class PaperRequest(BaseModel):
    text: str = Field(..., description="需要分析或总结的原始文本")
    style: str = Field(default="bullet", description="输出风格，例如 bullet / paragraph")
    max_points: int = Field(default=5, ge=1, le=10, description="最多返回几点，范围 1~10")


@app.get("/")
def read_root():
    return {
        "project_name": "ai-llm-learning",
        "version": "0.3.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"


@app.get("/echo/{name}")
def echo_name(name: str):
    return {"message": f"hello, {name}"}


@app.get("/summary")
def summary(text: str, style: str = "bullet", note: Optional[str] = None):
    return {
        "text": text,
        "style": style,
        "note": note,
        "summary": f"模拟摘要：text={text}, style={style}"
    }


@app.post("/analyze")
def analyze_paper(request: PaperRequest):
    return {
        "received_text": request.text,
        "received_style": request.style,
        "received_max_points": request.max_points,
        "message": "已成功收到 JSON body"
    }
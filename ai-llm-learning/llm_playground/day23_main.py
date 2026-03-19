from typing import Optional
from fastapi import FastAPI

app = FastAPI(
    title="AI LLM Learning API",
    version="0.2.0"
)


@app.get("/")
def read_root():
    return {
        "project_name": "ai-llm-learning",
        "version": "0.2.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"


@app.get("/echo/{name}")
def echo_name(name: str):
    return {
        "message": f"hello, {name}"
    }


@app.get("/summary")
def summary(text: str, style: str = "bullet", note: Optional[str] = None):
    return {
        "received_text": text,
        "style": style,
        "note": note,
        "summary": f"这是一个模拟摘要结果，当前 style={style}"
    }
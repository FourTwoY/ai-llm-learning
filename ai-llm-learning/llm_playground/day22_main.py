from fastapi import FastAPI

app = FastAPI(
    title="AI LLM Learning API",
    version="0.1.0"
)


@app.get("/")
def read_root():
    return {
        "project_name": "ai-llm-learning",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/ping")
def ping():
    return "pong"
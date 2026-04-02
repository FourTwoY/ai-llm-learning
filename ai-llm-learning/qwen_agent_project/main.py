from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from agent_api_service import run_agent_with_trace, run_debug_tool_case
except ModuleNotFoundError:
    from qwen_agent_project.agent_api_service import run_agent_with_trace, run_debug_tool_case


app = FastAPI(
    title="Day 34 Minimal Agent API",
    version="0.1.0",
    description="Minimal FastAPI wrapper for the qwen_agent_project agent loop.",
)


class AgentChatRequest(BaseModel):
    query: str = Field(..., description="用户输入的问题")
    max_tool_rounds: int = Field(3, ge=1, le=10, description="最大工具调用轮数")
    debug_tool_case: str | None = Field(
        default=None,
        description="可选调试场景：empty_args、internal_error、empty_result",
    )


class ToolTraceItem(BaseModel):
    round: int
    tool_name: str
    tool_args: dict[str, Any]
    success: bool
    error_message: str | None = None


class ToolOutputSummaryItem(BaseModel):
    round: int
    tool_name: str
    summary: str


class AgentChatResponse(BaseModel):
    final_answer: str
    tool_call_trace: list[ToolTraceItem]
    tool_outputs_summary: list[ToolOutputSummaryItem]


@app.get("/")
def health_check() -> dict[str, str]:
    return {"message": "Agent API is running. Open /docs to test /agent/chat."}


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
    if payload.debug_tool_case:
        response = run_debug_tool_case(payload.debug_tool_case)
    else:
        response = run_agent_with_trace(
            user_query=payload.query,
            max_tool_rounds=payload.max_tool_rounds,
            verbose=False,
        )
    return AgentChatResponse(**response)

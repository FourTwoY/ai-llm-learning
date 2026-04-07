from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    input: str = ""


class PredictData(BaseModel):
    answer: str
    model: str


class PredictResponse(BaseModel):
    success: bool
    message: str
    data: PredictData | None = None


class HealthResponse(BaseModel):
    success: bool
    message: str
    data: dict

from __future__ import annotations

try:
    from fastapi import FastAPI
except Exception:
    class FastAPI:
        def __init__(self, title: str = "", version: str = ""):
            self.title = title
            self.version = version
            self.routes = {}

        def get(self, path: str, response_model=None):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func

            return decorator

        def post(self, path: str, response_model=None):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

from lora_domain_learning_assistant.src.api.schemas import HealthResponse, PredictData, PredictRequest, PredictResponse
from lora_domain_learning_assistant.src.inference.predictor import DomainLearningPredictor
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.logger import logger

cfg = load_config()
app = FastAPI(title=cfg["api"]["title"], version=cfg["api"]["version"])
predictor = DomainLearningPredictor(lazy_load=True, enable_fallback=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(success=True, message="service is healthy", data={"model": predictor.model_name})


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result = predictor.predict(request.instruction, request.input)
        return PredictResponse(success=True, message="ok", data=PredictData(**result))
    except Exception as exc:
        logger.exception("Predict failed")
        return PredictResponse(success=False, message=str(exc), data=None)

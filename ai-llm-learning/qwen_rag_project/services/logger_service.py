import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get("-")
        return True


def setup_logger(name: str = "qwen_rag_project") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.addFilter(RequestIdFilter())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | request_id=%(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()


def set_request_id(request_id: str | None = None) -> str:
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return request_id_var.get("-")


def safe_preview(value, max_len: int = 120) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


@contextmanager
def log_step(step_name: str, **kwargs):
    """
    统一记录：
    - 开始时间
    - 输入摘要
    - 成功/失败
    - 返回条数
    - 耗时
    """
    start = time.perf_counter()
    input_summary = {k: safe_preview(v) for k, v in kwargs.items()}

    logger.info(f"[{step_name}] START | input={input_summary}")
    try:
        yield
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(f"[{step_name}] END | duration_ms={duration_ms}")
    except Exception as e:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(f"[{step_name}] ERROR | duration_ms={duration_ms} | error={e}")
        raise


def log_result(step_name: str, result_count: int | None = None, extra: dict | None = None):
    payload = {}
    if result_count is not None:
        payload["result_count"] = result_count
    if extra:
        payload.update(extra)
    logger.info(f"[{step_name}] RESULT | {payload}")
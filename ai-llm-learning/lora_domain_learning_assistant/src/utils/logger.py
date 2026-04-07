from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from time import perf_counter

from lora_domain_learning_assistant.src.utils.config import load_config


def setup_logger(name: str = "lora_domain_learning_assistant") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = str(load_config().get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()


@contextmanager
def log_step(step_name: str):
    start = perf_counter()
    logger.info("[%s] start", step_name)
    try:
        yield
        logger.info("[%s] done | elapsed_ms=%.2f", step_name, (perf_counter() - start) * 1000)
    except Exception:
        logger.exception("[%s] failed | elapsed_ms=%.2f", step_name, (perf_counter() - start) * 1000)
        raise

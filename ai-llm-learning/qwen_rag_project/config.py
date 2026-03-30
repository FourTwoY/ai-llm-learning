from __future__ import annotations
import copy
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "models": {
        "generation": "qwen3-max-2026-01-23",
        "embedding": "text-embedding-v4",
        "rerank": "qwen3-rerank",
        "rewrite": "qwen3-max-2026-01-23",
    },
    "chunking": {
        "chunk_size": 500,
        "overlap": 100,
    },
    "retrieval": {
        "top_k": 5,
        "rerank_top_n": 3,
        "use_hybrid": True,
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
    },
    "rewrite": {
        "use_rewrite": True,
    },
    "embedding": {
        "dimension": 1024,
        "batch_size": 10,
    },
    "logging": {
        "level": "INFO",
    },
    "paths": {
        "raw_dir": "data/raw",
        "processed_file": "data/processed/docs.json",
        "chunks_file": "data/chunks/chunks.json",
        "embeddings_file": "data/embeddings/all_embeddings.json",
    },
}

_CONFIG_CACHE: dict | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        base_value = base.get(key)
        if isinstance(base_value, dict):
            if not isinstance(value, dict):
                raise ValueError(f"config key '{key}' must be a mapping/object")
            _deep_merge(base_value, value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path = CONFIG_FILE) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    raw_text = path.read_text(encoding="utf-8")
    user_config = yaml.safe_load(raw_text) or {}

    if not isinstance(user_config, dict):
        raise ValueError("config.yaml 顶层必须是对象（key-value 结构）。")

    merged_config = copy.deepcopy(DEFAULT_CONFIG)
    return _deep_merge(merged_config, user_config)


def get_config(reload: bool = False) -> dict:
    global _CONFIG_CACHE
    if reload or _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE

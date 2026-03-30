from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.yaml"
DEFAULT_ENV_FILE = BASE_DIR / ".env"

DEFAULT_CONFIG = {
    "app": {
        "name": "qwen_rag_project",
        "env": "dev",
    },
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
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key": "",
    },
    "environments": {
        "dev": {},
        "prod": {},
    },
}

_CONFIG_CACHE: dict | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            _deep_merge(base_value, value)
        else:
            base[key] = value
    return base


def _load_yaml_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    raw_text = path.read_text(encoding="utf-8")
    user_config = yaml.safe_load(raw_text) or {}

    if not isinstance(user_config, dict):
        raise ValueError("config.yaml 顶层必须是对象（key-value 结构）。")

    return user_config


def _load_env_files(base_dir: Path, env_name: str) -> None:
    """
    加载环境变量：
    1. 先加载 .env
    2. 再加载 .env.dev / .env.prod（如果存在）
    后者可覆盖前者
    """
    default_env_file = base_dir / ".env"
    if default_env_file.exists():
        load_dotenv(default_env_file, override=False)

    env_specific_file = base_dir / f".env.{env_name}"
    if env_specific_file.exists():
        load_dotenv(env_specific_file, override=True)


def _resolve_runtime_env(config: dict) -> str:
    """
    APP_ENV 优先级最高，其次取 config.yaml 里的 app.env，默认 dev
    """
    return os.getenv("APP_ENV") or config.get("app", {}).get("env", "dev")


def _inject_secrets(config: dict) -> dict:
    api_key_env_name = config["dashscope"]["api_key_env"]
    api_key = os.getenv(api_key_env_name, "").strip()

    if not api_key:
        raise ValueError(
            f"缺少环境变量 {api_key_env_name}，请检查 .env / .env.dev / .env.prod 配置。"
        )

    config["dashscope"]["api_key"] = api_key
    return config


def load_config(config_path: str | Path = CONFIG_FILE) -> dict:
    user_config = _load_yaml_config(config_path)

    merged_config = copy.deepcopy(DEFAULT_CONFIG)
    _deep_merge(merged_config, user_config)

    env_name = _resolve_runtime_env(merged_config)
    _load_env_files(BASE_DIR, env_name)

    # 再次确定最终 env
    env_name = _resolve_runtime_env(merged_config)
    merged_config["app"]["env"] = env_name

    env_overrides = merged_config.get("environments", {}).get(env_name, {})
    if not isinstance(env_overrides, dict):
        raise ValueError(f"config.environments.{env_name} 必须是对象。")

    _deep_merge(merged_config, env_overrides)
    _inject_secrets(merged_config)

    return merged_config


def get_config(reload: bool = False) -> dict:
    global _CONFIG_CACHE
    if reload or _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
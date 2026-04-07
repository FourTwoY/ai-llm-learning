from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_dir": str(PROJECT_ROOT / "outputs" / "lora_adapter"),
        "max_seq_length": 512,
        "use_4bit": False,
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
    },
    "data": {
        "raw_dir": str(PROJECT_ROOT.parent / "qwen_rag_project" / "data" / "raw"),
        "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
        "train_file": str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"),
        "val_file": str(PROJECT_ROOT / "data" / "processed" / "val.jsonl"),
        "eval_file": str(PROJECT_ROOT / "data" / "processed" / "eval.jsonl"),
        "sample_preview_file": str(PROJECT_ROOT / "data" / "processed" / "sample_preview.md"),
        "min_output_chars": 20,
        "max_output_chars": 1200,
        "duplicate_jaccard_threshold": 0.92,
        "train_size": 120,
        "val_size": 24,
        "eval_size": 24,
        "random_seed": 42,
    },
    "lora": {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "training": {
        "output_dir": str(PROJECT_ROOT / "outputs" / "sft_lora"),
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "logging_steps": 5,
        "save_steps": 20,
        "eval_steps": 20,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": False,
        "smoke_max_train_samples": 8,
        "smoke_max_eval_samples": 4,
    },
    "inference": {
        "output_file": str(PROJECT_ROOT / "reports" / "inference_results.jsonl"),
        "batch_input_file": str(PROJECT_ROOT / "data" / "processed" / "eval.jsonl"),
        "prefer_adapter": True,
    },
    "evaluation": {
        "report_file": str(PROJECT_ROOT / "reports" / "experiment_report.md"),
        "compare_file": str(PROJECT_ROOT / "reports" / "before_after_compare.md"),
        "max_eval_samples": 20,
    },
    "api": {"host": "127.0.0.1", "port": 8000, "title": "LoRA Domain Learning Assistant API", "version": "0.1.0"},
    "logging": {"level": "INFO"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT.parent / path).resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    env_config = os.getenv("LORA_ASSISTANT_CONFIG")
    config_file = Path(config_path or env_config or DEFAULT_CONFIG_PATH)
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT.parent / config_file).resolve()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_file.exists():
        user_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError(f"配置文件顶层必须是字典: {config_file}")
        _deep_merge(cfg, user_cfg)

    path_keys = {
        "model": ["adapter_dir"],
        "data": ["raw_dir", "processed_dir", "train_file", "val_file", "eval_file", "sample_preview_file"],
        "training": ["output_dir"],
        "inference": ["output_file", "batch_input_file"],
        "evaluation": ["report_file", "compare_file"],
    }
    for section, keys in path_keys.items():
        for key in keys:
            cfg[section][key] = str(resolve_path(cfg[section][key]))
    return cfg

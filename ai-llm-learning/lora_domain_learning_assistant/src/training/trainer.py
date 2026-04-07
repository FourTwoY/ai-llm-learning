from __future__ import annotations

from pathlib import Path

from lora_domain_learning_assistant.src.dataset.formatter import format_sft_text
from lora_domain_learning_assistant.src.dataset.loader import load_sft_records
from lora_domain_learning_assistant.src.training.lora_config import build_lora_config
from lora_domain_learning_assistant.src.training.model_loader import load_base_model, load_tokenizer
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.io import ensure_parent, write_text_file
from lora_domain_learning_assistant.src.utils.logger import logger


def _build_training_arguments(cfg: dict):
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    output_dir = ensure_parent(Path(train_cfg["output_dir"]) / "trainer_state.json").parent
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=float(train_cfg["weight_decay"]),
        fp16=bool(train_cfg["fp16"]),
        bf16=bool(train_cfg["bf16"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        report_to=[],
        save_total_limit=2,
        evaluation_strategy="steps",
    )


def _to_dataset(records: list[dict], max_samples: int | None = None):
    from datasets import Dataset

    selected = records[:max_samples] if max_samples else records
    return Dataset.from_list([{"text": format_sft_text(item)} for item in selected])


def train_lora(config_path: str | Path | None = None, smoke_test: bool = False) -> dict:
    from trl import SFTTrainer

    cfg = load_config(config_path)
    tokenizer = load_tokenizer(cfg["model"]["base_model"], cfg["model"].get("trust_remote_code", True))
    model = load_base_model(cfg)
    peft_config = build_lora_config(cfg)

    train_records = load_sft_records(cfg["data"]["train_file"])
    val_records = load_sft_records(cfg["data"]["val_file"])
    if not train_records or not val_records:
        raise ValueError("训练/验证数据为空，请先运行 scripts/prepare_data.py")

    train_max = cfg["training"]["smoke_max_train_samples"] if smoke_test else None
    eval_max = cfg["training"]["smoke_max_eval_samples"] if smoke_test else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_to_dataset(train_records, train_max),
        eval_dataset=_to_dataset(val_records, eval_max),
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=int(cfg["model"]["max_seq_length"]),
        args=_build_training_arguments(cfg),
    )

    logger.info("Start SFT training | smoke_test=%s", smoke_test)
    trainer.train()

    adapter_dir = Path(cfg["model"]["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if config_path and Path(config_path).exists():
        write_text_file(adapter_dir / "training_config_snapshot.yaml", Path(config_path).read_text(encoding="utf-8"))

    metrics = trainer.evaluate()
    logger.info("Training finished | metrics=%s", metrics)
    return {"adapter_dir": str(adapter_dir), "metrics": metrics, "output_dir": cfg["training"]["output_dir"]}

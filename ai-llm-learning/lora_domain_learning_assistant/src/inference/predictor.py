from __future__ import annotations

from pathlib import Path

from lora_domain_learning_assistant.src.inference.generator import fallback_answer, generate_answer
from lora_domain_learning_assistant.src.training.model_loader import load_model_with_optional_adapter
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.logger import logger


class DomainLearningPredictor:
    def __init__(
        self,
        config_path: str | Path | None = None,
        adapter_dir: str | Path | None = None,
        lazy_load: bool = True,
        enable_fallback: bool = True,
    ):
        self.cfg = load_config(config_path)
        self.adapter_dir = adapter_dir if adapter_dir is not None else self.cfg["model"]["adapter_dir"]
        self.enable_fallback = enable_fallback
        self.tokenizer = None
        self.model = None
        self.model_name = self.cfg["model"]["base_model"]
        self._load_attempted = False
        if not lazy_load:
            self.load()

    def load(self):
        if self._load_attempted and (self.tokenizer is None or self.model is None):
            return
        self._load_attempted = True
        try:
            self.tokenizer, self.model = load_model_with_optional_adapter(self.cfg, self.adapter_dir)
            adapter_path = Path(self.adapter_dir) if self.adapter_dir else None
            if adapter_path and adapter_path.exists() and any(adapter_path.iterdir()):
                self.model_name = f"{self.cfg['model']['base_model']} + LoRA"
            else:
                self.model_name = self.cfg["model"]["base_model"]
        except Exception as exc:
            if not self.enable_fallback:
                raise
            logger.warning("模型加载失败，启用规则化 fallback 推理: %s", exc)
            self.tokenizer = None
            self.model = None
            self.model_name = "fallback-rule-based"

    def predict(self, instruction: str, input_text: str = "") -> dict:
        instruction = str(instruction or "").strip()
        input_text = str(input_text or "").strip()
        if not instruction:
            raise ValueError("instruction 不能为空")

        if self.tokenizer is None or self.model is None:
            self.load()

        if self.tokenizer is None or self.model is None:
            return {"answer": fallback_answer(instruction, input_text), "model": self.model_name}

        answer = generate_answer(
            self.tokenizer,
            self.model,
            instruction,
            input_text,
            max_new_tokens=int(self.cfg["model"]["max_new_tokens"]),
            temperature=float(self.cfg["model"]["temperature"]),
            top_p=float(self.cfg["model"]["top_p"]),
            do_sample=bool(self.cfg["model"]["do_sample"]),
        )
        return {"answer": answer, "model": self.model_name}

    def batch_predict(self, records: list[dict]) -> list[dict]:
        return [
            {**record, **self.predict(record.get("instruction", ""), record.get("input", ""))}
            for record in records
        ]

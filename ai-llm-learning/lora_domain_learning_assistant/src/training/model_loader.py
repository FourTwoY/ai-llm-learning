from __future__ import annotations

from pathlib import Path

from lora_domain_learning_assistant.src.utils.logger import logger


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_tokenizer(base_model: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM

    model_cfg = cfg["model"]
    kwargs = {
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "device_map": model_cfg.get("device_map", "auto"),
    }
    use_4bit = bool(model_cfg.get("use_4bit", False))

    if use_4bit and _bitsandbytes_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading base model with 4-bit QLoRA enabled.")
    else:
        if use_4bit:
            logger.warning("bitsandbytes 不可用，自动回退到普通 LoRA 加载。")
        torch_dtype = model_cfg.get("torch_dtype", "auto")
        if torch_dtype != "auto" and hasattr(torch, str(torch_dtype)):
            kwargs["torch_dtype"] = getattr(torch, str(torch_dtype))

    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def load_model_with_optional_adapter(cfg: dict, adapter_dir: str | Path | None = None):
    from peft import PeftModel

    tokenizer = load_tokenizer(cfg["model"]["base_model"], cfg["model"].get("trust_remote_code", True))
    model = load_base_model(cfg)
    adapter_path = Path(adapter_dir or cfg["model"]["adapter_dir"])
    if adapter_path.exists() and any(adapter_path.iterdir()):
        model = PeftModel.from_pretrained(model, str(adapter_path))
        logger.info("Loaded LoRA adapter from %s", adapter_path)
    else:
        logger.info("Adapter not found, using base model only: %s", adapter_path)
    model.eval()
    return tokenizer, model

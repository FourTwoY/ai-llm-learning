from __future__ import annotations

def build_lora_config(cfg: dict):
    from peft import LoraConfig

    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["lora_r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )

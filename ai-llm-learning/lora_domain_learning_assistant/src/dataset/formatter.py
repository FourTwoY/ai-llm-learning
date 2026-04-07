from __future__ import annotations

SYSTEM_PROMPT = "你是一个面向 AI 论文和机器学习知识的领域学习助手，请用清晰、简洁、稳定的中文回答。"


def normalize_sft_record(record: dict) -> dict:
    return {
        "instruction": str(record.get("instruction", "")).strip(),
        "input": str(record.get("input", "")).strip(),
        "output": str(record.get("output", "")).strip(),
    }


def format_prompt(instruction: str, input_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"### 指令\n{instruction}\n\n"
            f"### 输入\n{input_text}\n\n"
            f"### 回答\n"
        )
    return f"{SYSTEM_PROMPT}\n\n### 指令\n{instruction}\n\n### 回答\n"


def format_sft_text(record: dict) -> str:
    item = normalize_sft_record(record)
    return format_prompt(item["instruction"], item["input"]) + item["output"]

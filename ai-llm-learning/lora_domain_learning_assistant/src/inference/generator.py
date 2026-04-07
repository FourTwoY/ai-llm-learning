from __future__ import annotations

from lora_domain_learning_assistant.src.dataset.formatter import format_prompt


def _strip_prompt(decoded_text: str, prompt: str) -> str:
    if decoded_text.startswith(prompt):
        decoded_text = decoded_text[len(prompt) :]
    return decoded_text.strip()


def generate_answer(
    tokenizer,
    model,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _strip_prompt(decoded_text, prompt) or decoded_text.strip()


def fallback_answer(instruction: str, input_text: str) -> str:
    prefix = "根据当前资料，" if input_text.strip() else ""
    return (
        f"{prefix}建议围绕“问题背景-核心方法-关键结论-适用边界”四点来回答：{instruction.strip()}。"
        "如果你正在做学习笔记，可以先提炼术语定义，再补充方法步骤和复习建议。"
    )

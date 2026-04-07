from __future__ import annotations

import random
import re
from pathlib import Path

from lora_domain_learning_assistant.src.dataset.loader import load_raw_documents
from lora_domain_learning_assistant.src.dataset.quality_check import audit_documents, validate_sft_records
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.io import write_jsonl, write_text_file


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_markdown_document(document: dict) -> dict:
    text = _clean_text(document["text"])
    lines = [line.strip().lstrip("\ufeff\u200b") for line in text.splitlines() if line.strip()]
    title = document["doc_id"].strip()
    sections: dict[str, list[str]] = {}
    current = "正文"

    for line in lines:
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line).strip()
            title_candidate = heading.replace("论文标题：", "").strip()
            if title == document["doc_id"] and title_candidate and not re.match(r"^\d+[\.\uff0e、\s]", title_candidate):
                title = title_candidate
            current = heading
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(re.sub(r"^[-*]\s*", "", line).strip())

    merged_sections = {
        section: "\n".join(content).strip()
        for section, content in sections.items()
        if "\n".join(content).strip()
    }
    summary_parts = [
        text
        for section_name, text in merged_sections.items()
        if any(key in section_name for key in ["摘要", "核心", "方法", "结论", "贡献", "总结"])
    ]
    summary = _clean_text("\n".join(summary_parts) or "\n".join(list(merged_sections.values())[:3]))
    concepts = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title) or [title[:40]]

    return {
        "doc_id": document["doc_id"],
        "source": document["source"],
        "title": title,
        "sections": merged_sections,
        "summary": summary,
        "concepts": concepts[:6],
        "raw_text": text,
    }


def build_instruction_records(parsed_docs: list[dict]) -> list[dict]:
    records = []
    for doc in parsed_docs:
        title = doc["title"]
        summary = doc["summary"]
        section_names = [name for name in doc["sections"] if name != title]
        core_points = "；".join(
            doc["sections"][name].split("\n")[0][:180]
            for name in section_names[:3]
            if doc["sections"][name].strip()
        )
        concept = doc["concepts"][0]
        context = f"资料标题：{title}\n来源文件：{doc['source']}\n内容摘录：{summary[:500]}"

        records.extend(
            [
                {
                    "instruction": f"请用 3 点概括《{title}》的核心思想。",
                    "input": context,
                    "output": f"《{title}》的核心思想可以概括为：1）{summary[:180]}；2）关键方法线索包括 {core_points or concept}；3）学习时建议重点把握问题定义、方法创新点和适用边界。",
                },
                {
                    "instruction": f"请解释概念“{concept}”，并说明它和《{title}》的关系。",
                    "input": context,
                    "output": f"{concept} 可以理解为这篇资料中的关键概念或方法线索。结合《{title}》，它通常用于说明模型结构、训练目标或知识迁移方式。学习时建议先记住它解决的问题，再对照方法部分理解实现思路。",
                },
                {
                    "instruction": f"基于资料《{title}》，生成一个适合初学者的 FAQ 问答。",
                    "input": summary[:600],
                    "output": f"问：这篇资料《{title}》主要想解决什么问题？\n答：它主要围绕 {summary[:220]} 展开。对初学者来说，先理解任务背景，再看作者提出了哪些结构或训练改动，会更容易建立整体认识。",
                },
                {
                    "instruction": f"请从《{title}》中抽取 3 个值得复习的知识点。",
                    "input": context,
                    "output": f"建议重点复习这 3 个知识点：1）{concept} 的基本含义与适用场景；2）文档中的核心方法流程：{core_points or summary[:180]}；3）该方法相对已有方案的收益、限制与适合对比的基线。",
                },
                {
                    "instruction": f"请把《{title}》整理成一段简短教学式讲解。",
                    "input": summary[:600],
                    "output": f"可以这样理解《{title}》：先看它要解决的任务，再看它为什么需要新的模型设计或训练策略。文档主线是 {summary[:260]}。如果在做学习笔记，建议按“问题-方法-优点-局限”四项整理。",
                },
                {
                    "instruction": "请根据给定资料，输出稳定、简洁的学习卡片。",
                    "input": context,
                    "output": f"学习卡片\n主题：{title}\n关键词：{', '.join(doc['concepts'][:4])}\n核心内容：{summary[:220]}\n复习建议：先看背景和任务定义，再整理方法步骤，最后补充适用场景与局限。",
                },
                {
                    "instruction": f"如果要向同学介绍《{title}》，请给出一段 100 字以内的中文说明。",
                    "input": "",
                    "output": f"《{title}》是一篇围绕 {concept} 和相关模型/训练方法展开的学习资料，重点介绍 {summary[:180]}。适合快速理解该方向的核心问题、方法创新点和应用边界。",
                },
                {
                    "instruction": f"请基于《{title}》设计一个复习提问，并直接给出参考答案。",
                    "input": context,
                    "output": f"复习提问：阅读《{title}》时，最应该先抓住哪条主线？\n参考答案：优先抓住“要解决的问题是什么、核心方法如何改进、这些改进带来什么收益和限制”这条主线，并结合 {concept} 等关键词做归纳。",
                },
                {
                    "instruction": f"请提炼《{title}》中适合写入笔记的章节总结。",
                    "input": "\n".join(f"{name}: {text[:180]}" for name, text in list(doc["sections"].items())[:4]),
                    "output": f"章节总结建议：围绕《{title}》，先记录背景与任务设定，再总结方法结构和训练要点，随后补充实验结论。根据当前资料，可优先记下：{summary[:260]}",
                },
                {
                    "instruction": f"请判断《{title}》更适合作为哪类学习资料，并说明理由。",
                    "input": summary[:500],
                    "output": f"《{title}》更适合作为“论文方法速读 + 概念复习”类学习资料，因为文本已经按标题、方法、结论等线索组织，适合抽取知识点、生成 FAQ 和做章节总结，但不适合原样当作对话式 SFT 答案直接训练。",
                },
            ]
        )
    return records


def split_records(records: list[dict], train_size: int, val_size: int, eval_size: int, seed: int) -> dict[str, list[dict]]:
    rnd = random.Random(seed)
    shuffled = records[:]
    rnd.shuffle(shuffled)
    total_needed = train_size + val_size + eval_size
    if len(shuffled) < total_needed:
        raise ValueError(f"样本数不足: 需要 {total_needed}，实际 {len(shuffled)}")
    return {
        "train": shuffled[:train_size],
        "val": shuffled[train_size : train_size + val_size],
        "eval": shuffled[train_size + val_size : train_size + val_size + eval_size],
    }


def render_sample_preview(split_data: dict[str, list[dict]], max_items: int = 5) -> str:
    lines = [
        "# Sample Preview",
        "",
        "这是一套用于跑通 LoRA/QLoRA 项目闭环的种子数据集，后续应继续人工审查、修订和扩充。",
        "",
    ]
    for split_name, records in split_data.items():
        lines.extend([f"## {split_name}", ""])
        for idx, record in enumerate(records[:max_items], start=1):
            lines.extend(
                [
                    f"### {split_name}-{idx}",
                    f"- instruction: {record['instruction']}",
                    f"- input: {record['input'][:220]}",
                    f"- output: {record['output'][:320]}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def generate_dataset(config_path: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    docs = load_raw_documents(cfg["data"]["raw_dir"])
    parsed_docs = [parse_markdown_document(doc) for doc in docs]
    records = build_instruction_records(parsed_docs)
    cleaned_records, stats = validate_sft_records(
        records,
        min_output_chars=cfg["data"]["min_output_chars"],
        max_output_chars=cfg["data"]["max_output_chars"],
        duplicate_jaccard_threshold=cfg["data"]["duplicate_jaccard_threshold"],
    )
    split_data = split_records(
        cleaned_records,
        cfg["data"]["train_size"],
        cfg["data"]["val_size"],
        cfg["data"]["eval_size"],
        cfg["data"]["random_seed"],
    )
    write_jsonl(cfg["data"]["train_file"], split_data["train"])
    write_jsonl(cfg["data"]["val_file"], split_data["val"])
    write_jsonl(cfg["data"]["eval_file"], split_data["eval"])
    write_text_file(cfg["data"]["sample_preview_file"], render_sample_preview(split_data))
    return {
        "raw_audit": audit_documents(docs),
        "generation_stats": stats,
        "split_sizes": {key: len(value) for key, value in split_data.items()},
    }

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_domain_learning_assistant.src.dataset.loader import load_raw_documents
from lora_domain_learning_assistant.src.dataset.quality_check import audit_documents
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.io import write_text_file
from lora_domain_learning_assistant.src.utils.logger import log_step, logger


def render_audit_report(documents: list[dict], audit_result: dict) -> str:
    lines = [
        "# Raw Data Audit",
        "",
        "## 结论",
        "",
        "- `qwen_rag_project/data/raw` 中的原始文件主要是 Markdown 论文卡片，更适合作为“知识来源材料”，不适合直接当作 SFT 的 instruction/output 对话样本。",
        "- 主要原因：文本是结构化笔记而非问答格式；不同文件之间模板高度一致，容易产生风格单一和重复样本；控制台抽样可观察到中文字符渲染异常风险，需要先清洗再构造统一输出风格的数据。",
        "- 因此本项目采用“先审计原文、再抽取知识点、最后自动构造 FAQ/概念解释/章节总结/知识点抽取/教学式回答”的种子数据策略。",
        "",
        "## 数据概览",
        "",
        f"- 文档总数: {audit_result['total_docs']}",
        f"- 文件类型分布: {audit_result['suffix_counter']}",
        f"- 平均字符数: {audit_result['avg_chars']}",
        f"- 过短文档数(<200 chars): {len(audit_result['short_docs'])}",
        f"- 疑似编码异常文档数: {len(audit_result['broken_encoding_docs'])}",
        f"- 近重复文档对数(Jaccard>=0.9): {len(audit_result['near_duplicate_pairs'])}",
        "",
        "## 文件明细",
        "",
        "| source | suffix | chars | lines | direct_sft_ready |",
        "|---|---|---:|---:|---|",
    ]
    for doc in documents:
        lines.append(
            f"| {doc['source']} | {doc['suffix']} | {doc['char_count']} | {doc['line_count']} | 否，建议先转 instruction 数据 |"
        )

    lines.extend(["", "## 疑似问题", ""])
    if audit_result["broken_encoding_docs"]:
        lines.append("### 编码/渲染异常")
        for doc in audit_result["broken_encoding_docs"]:
            preview = doc["text"][:120].replace("\n", " ")
            lines.append(f"- {doc['source']}: {preview}")
        lines.append("")
    if audit_result["near_duplicate_pairs"]:
        lines.append("### 近重复文档对")
        for left, right, score in audit_result["near_duplicate_pairs"]:
            lines.append(f"- {left} <-> {right}: {score}")
        lines.append("")
    if not audit_result["broken_encoding_docs"] and not audit_result["near_duplicate_pairs"]:
        lines.extend(["- 未发现明显异常。", ""])

    lines.extend(
        [
            "## 建议",
            "",
            "- 不要直接把原始 Markdown 全文拼成 output 做 SFT。",
            "- 先抽取标题、摘要、方法、结论等稳定片段，再构造统一 schema 的 instruction 数据。",
            "- 本仓库生成的数据集定位为“用于跑通 LoRA/QLoRA 项目闭环的种子数据集”，后续应继续人工审查与扩充。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    report_file = PROJECT_ROOT / "reports" / "raw_data_audit.md"
    with log_step("audit_raw_data"):
        docs = load_raw_documents(cfg["data"]["raw_dir"])
        audit_result = audit_documents(docs)
        write_text_file(report_file, render_audit_report(docs, audit_result))
        logger.info("Audit report saved to %s", report_file)


if __name__ == "__main__":
    main()

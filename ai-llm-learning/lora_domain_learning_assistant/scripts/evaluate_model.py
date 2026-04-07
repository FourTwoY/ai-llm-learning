from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_domain_learning_assistant.src.dataset.loader import load_sft_records
from lora_domain_learning_assistant.src.inference.predictor import DomainLearningPredictor
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.io import write_text_file
from lora_domain_learning_assistant.src.utils.logger import log_step, logger
from lora_domain_learning_assistant.src.utils.metrics import keyword_overlap, rouge_l_like


def _compare_label(base_score: float, tuned_score: float) -> str:
    if tuned_score > base_score + 0.05:
        return "LoRA 更接近参考答案"
    if base_score > tuned_score + 0.05:
        return "Base 更接近参考答案"
    return "两者接近"


def render_compare_report(rows: list[dict]) -> str:
    lines = [
        "# Before/After Compare",
        "",
        "| # | instruction | reference | base_answer | lora_answer | compare |",
        "|---|---|---|---|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | {inst} | {ref} | {base} | {lora} | {compare} |".format(
                idx=idx,
                inst=row["instruction"].replace("|", "/")[:120],
                ref=row["reference"].replace("|", "/")[:180],
                base=row["base_answer"].replace("|", "/")[:180],
                lora=row["lora_answer"].replace("|", "/")[:180],
                compare=row["compare"],
            )
        )
    return "\n".join(lines) + "\n"


def render_experiment_report(rows: list[dict], cfg: dict) -> str:
    avg_base_rouge = sum(row["base_rouge_l"] for row in rows) / max(len(rows), 1)
    avg_lora_rouge = sum(row["lora_rouge_l"] for row in rows) / max(len(rows), 1)
    avg_base_kw = sum(row["base_keyword_overlap"] for row in rows) / max(len(rows), 1)
    avg_lora_kw = sum(row["lora_keyword_overlap"] for row in rows) / max(len(rows), 1)

    return "\n".join(
        [
            "# Experiment Report",
            "",
            "## 实验设置",
            "",
            f"- Base model: `{cfg['model']['base_model']}`",
            f"- Adapter dir: `{cfg['model']['adapter_dir']}`",
            f"- Eval file: `{cfg['data']['eval_file']}`",
            f"- Max eval samples: {cfg['evaluation']['max_eval_samples']}",
            "",
            "## 自动指标",
            "",
            "| model | avg_rouge_l_like | avg_keyword_overlap |",
            "|---|---:|---:|",
            f"| base | {avg_base_rouge:.4f} | {avg_base_kw:.4f} |",
            f"| lora | {avg_lora_rouge:.4f} | {avg_lora_kw:.4f} |",
            "",
            "## 解读",
            "",
            "- 该评测主要用于固定 eval 集上的“微调前后可读对比”，不是严格 benchmark。",
            "- 如果本地尚未训练出 adapter，LoRA 列可能自动回退到 base/fallback 行为，此时需要先运行 `scripts/train_lora.py`。",
            "- 当前数据集是自动构造的种子集，适合验证闭环，不代表最终高质量领域训练集上限。",
            "",
        ]
    )


def main() -> None:
    cfg = load_config()
    with log_step("evaluate_model"):
        eval_records = load_sft_records(cfg["data"]["eval_file"])[: cfg["evaluation"]["max_eval_samples"]]
        base_predictor = DomainLearningPredictor(adapter_dir="", lazy_load=True, enable_fallback=True)
        lora_predictor = DomainLearningPredictor(
            adapter_dir=cfg["model"]["adapter_dir"],
            lazy_load=True,
            enable_fallback=True,
        )

        rows = []
        for record in eval_records:
            base_result = base_predictor.predict(record["instruction"], record["input"])
            lora_result = lora_predictor.predict(record["instruction"], record["input"])
            base_rouge = rouge_l_like(record["output"], base_result["answer"])
            lora_rouge = rouge_l_like(record["output"], lora_result["answer"])
            base_kw = keyword_overlap(record["output"], base_result["answer"])
            lora_kw = keyword_overlap(record["output"], lora_result["answer"])
            rows.append(
                {
                    "instruction": record["instruction"],
                    "reference": record["output"],
                    "base_answer": base_result["answer"],
                    "lora_answer": lora_result["answer"],
                    "base_rouge_l": base_rouge,
                    "lora_rouge_l": lora_rouge,
                    "base_keyword_overlap": base_kw,
                    "lora_keyword_overlap": lora_kw,
                    "compare": _compare_label(base_rouge + base_kw, lora_rouge + lora_kw),
                }
            )

        write_text_file(cfg["evaluation"]["compare_file"], render_compare_report(rows))
        write_text_file(cfg["evaluation"]["report_file"], render_experiment_report(rows, cfg))
        logger.info("Evaluation reports saved.")


if __name__ == "__main__":
    main()

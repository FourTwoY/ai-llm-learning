from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_domain_learning_assistant.src.inference.predictor import DomainLearningPredictor
from lora_domain_learning_assistant.src.utils.io import write_text_file
from lora_domain_learning_assistant.src.utils.logger import log_step, logger


DEMO_QUESTIONS = [
    {"instruction": "请用 3 点解释 LoRA 的核心思想。", "input": "面向刚入门微调的新同学，回答要简洁。"},
    {"instruction": "什么是 RAG？它适合解决哪类问题？", "input": "请用学习笔记风格回答。"},
    {"instruction": "请把 Transformer 自注意力机制讲成一段适合复习的短说明。", "input": ""},
]


def main() -> None:
    predictor = DomainLearningPredictor(lazy_load=True, enable_fallback=True)
    lines = ["# Demo Examples", ""]
    with log_step("build_demo_examples"):
        for idx, item in enumerate(DEMO_QUESTIONS, start=1):
            result = predictor.predict(item["instruction"], item["input"])
            lines.extend(
                [
                    f"## 示例 {idx}",
                    "",
                    f"- instruction: {item['instruction']}",
                    f"- input: {item['input']}",
                    f"- model: {result['model']}",
                    f"- answer: {result['answer']}",
                    "",
                ]
            )
        output_path = PROJECT_ROOT / "reports" / "demo_examples.md"
        write_text_file(output_path, "\n".join(lines))
        logger.info("Demo examples saved to %s", output_path)


if __name__ == "__main__":
    main()

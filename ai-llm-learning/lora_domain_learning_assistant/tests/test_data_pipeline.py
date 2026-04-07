from lora_domain_learning_assistant.src.dataset.formatter import format_prompt, format_sft_text
from lora_domain_learning_assistant.src.dataset.generator import parse_markdown_document
from lora_domain_learning_assistant.src.dataset.quality_check import validate_sft_records


def test_format_prompt_and_sft_text_keep_schema_stable():
    record = {"instruction": "解释 LoRA", "input": "面向初学者", "output": "LoRA 是低秩适配方法。"}

    prompt = format_prompt(record["instruction"], record["input"])
    sft_text = format_sft_text(record)

    assert "### 指令" in prompt
    assert "### 输入" in prompt
    assert sft_text.endswith("LoRA 是低秩适配方法。")


def test_parse_markdown_document_extracts_title_and_sections():
    document = {
        "doc_id": "lora",
        "source": "lora.md",
        "text": "# 论文标题：LoRA Low-Rank Adaptation\n\n## 摘要\nLoRA 用低秩矩阵做高效微调。\n\n## 方法\n冻结基座参数，仅训练适配器。",
    }

    parsed = parse_markdown_document(document)

    assert "LoRA" in parsed["title"]
    assert parsed["summary"]
    assert parsed["sections"]


def test_validate_sft_records_drops_bad_and_duplicate_records():
    records = [
        {"instruction": "解释 LoRA", "input": "", "output": "LoRA 通过低秩矩阵实现参数高效微调。"},
        {"instruction": "解释 LoRA", "input": "", "output": "LoRA 通过低秩矩阵实现参数高效微调。"},
        {"instruction": "", "input": "", "output": "无效"},
        {"instruction": "太短", "input": "", "output": "短"},
    ]

    cleaned, stats = validate_sft_records(records, min_output_chars=8, max_output_chars=200, duplicate_jaccard_threshold=0.9)

    assert len(cleaned) == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_empty_field"] == 1
    assert stats["dropped_short_output"] == 1

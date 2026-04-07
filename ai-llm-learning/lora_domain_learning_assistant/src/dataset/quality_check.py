from __future__ import annotations

from collections import Counter

from lora_domain_learning_assistant.src.utils.metrics import jaccard_similarity


def audit_documents(documents: list[dict]) -> dict:
    suffix_counter = Counter(doc["suffix"] for doc in documents)
    short_docs = [doc for doc in documents if doc["char_count"] < 200]
    broken_encoding_docs = [
        doc for doc in documents if "锛" in doc["text"] or "绛" in doc["text"] or "鍩" in doc["text"]
    ]
    near_duplicate_pairs = []
    for idx, left in enumerate(documents):
        for right in documents[idx + 1 :]:
            score = jaccard_similarity(left["text"], right["text"])
            if score >= 0.9:
                near_duplicate_pairs.append((left["source"], right["source"], round(score, 4)))

    return {
        "total_docs": len(documents),
        "suffix_counter": dict(suffix_counter),
        "short_docs": short_docs,
        "broken_encoding_docs": broken_encoding_docs,
        "near_duplicate_pairs": near_duplicate_pairs,
        "avg_chars": round(sum(doc["char_count"] for doc in documents) / max(len(documents), 1), 2),
    }


def validate_sft_records(
    records: list[dict],
    min_output_chars: int,
    max_output_chars: int,
    duplicate_jaccard_threshold: float,
) -> tuple[list[dict], dict]:
    cleaned = []
    stats = {
        "input_records": len(records),
        "kept_records": 0,
        "dropped_short_output": 0,
        "dropped_long_output": 0,
        "dropped_duplicate": 0,
        "dropped_empty_field": 0,
    }

    for record in records:
        instruction = str(record.get("instruction", "")).strip()
        input_text = str(record.get("input", "")).strip()
        output = str(record.get("output", "")).strip()
        if not instruction or not output:
            stats["dropped_empty_field"] += 1
            continue
        if len(output) < min_output_chars:
            stats["dropped_short_output"] += 1
            continue
        if len(output) > max_output_chars:
            stats["dropped_long_output"] += 1
            continue

        merged_text = f"{instruction}\n{input_text}\n{output}"
        if any(
            jaccard_similarity(merged_text, f"{item['instruction']}\n{item['input']}\n{item['output']}")
            >= duplicate_jaccard_threshold
            for item in cleaned
        ):
            stats["dropped_duplicate"] += 1
            continue
        cleaned.append({"instruction": instruction, "input": input_text, "output": output})

    stats["kept_records"] = len(cleaned)
    return cleaned, stats

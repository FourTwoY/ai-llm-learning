from __future__ import annotations

import re
from collections import Counter


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def jaccard_similarity(left: str, right: str) -> float:
    left_set = set(tokenize_text(left))
    right_set = set(tokenize_text(right))
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def rouge_l_like(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    dp = [0] * (len(pred_tokens) + 1)
    for token in ref_tokens:
        prev = 0
        for idx, pred_token in enumerate(pred_tokens, start=1):
            cur = dp[idx]
            dp[idx] = prev + 1 if token == pred_token else max(dp[idx], dp[idx - 1])
            prev = cur

    lcs = dp[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def keyword_overlap(reference: str, prediction: str, top_k: int = 8) -> float:
    ref_counter = Counter(tokenize_text(reference))
    if not ref_counter:
        return 0.0
    keywords = {token for token, _ in ref_counter.most_common(top_k)}
    return len(keywords & set(tokenize_text(prediction))) / len(keywords)

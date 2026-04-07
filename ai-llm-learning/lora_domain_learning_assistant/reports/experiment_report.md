# Experiment Report

## 实验设置

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Adapter dir: `D:\PycharmProjects\pythonProject\ai-llm-learning\lora_domain_learning_assistant\outputs\lora_adapter`
- Eval file: `D:\PycharmProjects\pythonProject\ai-llm-learning\lora_domain_learning_assistant\data\processed\eval.jsonl`
- Max eval samples: 20

## 自动指标

| model | avg_rouge_l_like | avg_keyword_overlap |
|---|---:|---:|
| base | 0.2819 | 0.5625 |
| lora | 0.2819 | 0.5625 |

## 解读

- 该评测主要用于固定 eval 集上的“微调前后可读对比”，不是严格 benchmark。
- 如果本地尚未训练出 adapter，LoRA 列可能自动回退到 base/fallback 行为，此时需要先运行 `scripts/train_lora.py`。
- 当前数据集是自动构造的种子集，适合验证闭环，不代表最终高质量领域训练集上限。

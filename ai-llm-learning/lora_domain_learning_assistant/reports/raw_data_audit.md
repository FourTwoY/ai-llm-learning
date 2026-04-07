# Raw Data Audit

## 结论

- `qwen_rag_project/data/raw` 中的原始文件主要是 Markdown 论文卡片，更适合作为“知识来源材料”，不适合直接当作 SFT 的 instruction/output 对话样本。
- 主要原因：文本是结构化笔记而非问答格式；不同文件之间模板高度一致，容易产生风格单一和重复样本；控制台抽样可观察到中文字符渲染异常风险，需要先清洗再构造统一输出风格的数据。
- 因此本项目采用“先审计原文、再抽取知识点、最后自动构造 FAQ/概念解释/章节总结/知识点抽取/教学式回答”的种子数据策略。

## 数据概览

- 文档总数: 17
- 文件类型分布: {'.md': 17}
- 平均字符数: 1519.18
- 过短文档数(<200 chars): 0
- 疑似编码异常文档数: 0
- 近重复文档对数(Jaccard>=0.9): 0

## 文件明细

| source | suffix | chars | lines | direct_sft_ready |
|---|---|---:|---:|---|
| An Image is Worth 16x16 Words Transformers for Image Recognition at Scale.md | .md | 1645 | 39 | 否，建议先转 instruction 数据 |
| Attention Is All You Need.md | .md | 1811 | 39 | 否，建议先转 instruction 数据 |
| BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.md | .md | 1562 | 39 | 否，建议先转 instruction 数据 |
| BLIP Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.md | .md | 1775 | 55 | 否，建议先转 instruction 数据 |
| Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.md | .md | 1525 | 43 | 否，建议先转 instruction 数据 |
| Language Models are Few-Shot Learners.md | .md | 1270 | 42 | 否，建议先转 instruction 数据 |
| Learning Transferable Visual Models From Natural Language Supervision.md | .md | 1435 | 43 | 否，建议先转 instruction 数据 |
| LLaMA Open and Efficient Foundation Language Models.md | .md | 1516 | 43 | 否，建议先转 instruction 数据 |
| LoRA Low-Rank Adaptation of Large Language Models.md | .md | 1311 | 43 | 否，建议先转 instruction 数据 |
| Masked Autoencoders Are Scalable Vision Learners.md | .md | 1563 | 39 | 否，建议先转 instruction 数据 |
| Megatron-LM Training Multi-Billion Parameter Language Models Using Model Parallelism.md | .md | 1363 | 41 | 否，建议先转 instruction 数据 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md | .md | 1369 | 43 | 否，建议先转 instruction 数据 |
| RoBERTa A Robustly Optimized BERT Pretraining Approach.md | .md | 1507 | 43 | 否，建议先转 instruction 数据 |
| Scaling Laws for Neural Language Models.md | .md | 1427 | 43 | 否，建议先转 instruction 数据 |
| Swin Transformer Hierarchical Vision Transformer using Shifted Windows.md | .md | 1680 | 43 | 否，建议先转 instruction 数据 |
| Training Compute-Optimal Large Language Models.md | .md | 1433 | 43 | 否，建议先转 instruction 数据 |
| Training data-efficient image transformers & distillation through attention.md | .md | 1634 | 43 | 否，建议先转 instruction 数据 |

## 疑似问题

- 未发现明显异常。

## 建议

- 不要直接把原始 Markdown 全文拼成 output 做 SFT。
- 先抽取标题、摘要、方法、结论等稳定片段，再构造统一 schema 的 instruction 数据。
- 本仓库生成的数据集定位为“用于跑通 LoRA/QLoRA 项目闭环的种子数据集”，后续应继续人工审查与扩充。

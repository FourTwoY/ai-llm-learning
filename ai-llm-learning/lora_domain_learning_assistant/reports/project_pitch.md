# Project Pitch

## 一句话介绍

`lora_domain_learning_assistant` 是一个面向“论文/学习资料问答与讲解”场景的 LoRA/QLoRA 最小可用微调项目，覆盖原始数据审计、instruction 数据自动构造、SFT 训练、推理、评测和 FastAPI 服务。

## 为什么值得做

- RAG 适合“外部知识检索 + 生成”，但如果希望模型逐渐形成稳定教学风格、固定回答结构和领域表达习惯，就需要一套可控的 SFT/LoRA 闭环。
- 仓库已有 `qwen_rag_project/data/raw` 论文笔记语料，但它更像知识源文档，不是对话式训练样本。本项目补上的正是“从知识文档到可训练 instruction 数据”的工程链路。

## 当前 MVP 交付

- 原始数据审计报告
- 自动 instruction 数据构造脚本
- LoRA/QLoRA 训练脚本
- Base/LoRA 推理与对比评测脚本
- FastAPI 推理接口
- demo 样例生成脚本
- 基础 pytest 测试

## 后续演进建议

- 引入人工审核后的高质量领域 QA 数据，替换当前自动种子集。
- 增加真实训练后的响应偏好评测、事实一致性检查和拒答边界测试。
- 将 RAG 检索上下文与 LoRA 学到的回答风格结合，形成“知识可更新 + 输出风格稳定”的混合方案。

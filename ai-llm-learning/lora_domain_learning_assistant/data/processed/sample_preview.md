# Sample Preview

这是一套用于跑通 LoRA/QLoRA 项目闭环的种子数据集，后续应继续人工审查、修订和扩充。

## train

### train-1
- instruction: 请提炼《Attention Is All You Need》中适合写入笔记的章节总结。
- input: 1. 基本信息: **作者**：Ashish Vaswani 等 (Google Brain / Google Research)
**年份**：2017
**会议/期刊**：NeurIPS (Conference on Neural Information Processing Systems)
**研究方向**：自然语言处理 (NLP)、深度学习、序列建模、机器翻译
2. 这篇论文主要解决什么问题: 解决传统序列转导模型（如 RNN
- output: 章节总结建议：围绕《Attention Is All You Need》，先记录背景与任务设定，再总结方法结构和训练要点，随后补充实验结论。根据当前资料，可优先记下：抛弃循环（Recurrence）和卷积（Convolution），仅使用注意力机制（Attention Mechanism）来实现序列数据的建模。提出"Self-Attention"概念，让模型直接关注全局信息，实现任意位置间的直接交互。
采用标准的 Encoder-Decoder 架构。Encoder 和 Decoder 均由 N 个相同层堆叠而成。每层包含两个子层：多头注意力机制（Multi-Head Attention）和位置全连接前馈网络（Position-wi

### train-2
- instruction: 请基于《RoBERTa: A Robustly Optimized BERT Pretraining Approach》设计一个复习提问，并直接给出参考答案。
- input: 资料标题：RoBERTa: A Robustly Optimized BERT Pretraining Approach
来源文件：RoBERTa A Robustly Optimized BERT Pretraining Approach.md
内容摘录：重新评估并优化 BERT 的预训练超参数和策略。证明在相同架构下，通过更严谨的训练设计（如去除 NSP、动态掩码、更大批量）可以显著提升模型表现，强调工程优化的重要性。
架构与 BE
- output: 复习提问：阅读《RoBERTa: A Robustly Optimized BERT Pretraining Approach》时，最应该先抓住哪条主线？
参考答案：优先抓住“要解决的问题是什么、核心方法如何改进、这些改进带来什么收益和限制”这条主线，并结合 RoBERTa 等关键词做归纳。

### train-3
- instruction: 请解释概念“Megatron-LM”，并说明它和《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》的关系。
- input: 资料标题：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
来源文件：Megatron-LM Training Multi-Billion Parameter Language Models Using Model Parallelism.md
内容摘录：采用模型并行（Model Parallelism）策略，特别是
- output: Megatron-LM 可以理解为这篇资料中的关键概念或方法线索。结合《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》，它通常用于说明模型结构、训练目标或知识迁移方式。学习时建议先记住它解决的问题，再对照方法部分理解实现思路。

### train-4
- instruction: 请把《Language Models are Few-Shot Learners》整理成一段简短教学式讲解。
- input: 提出“少样本学习”（Few-Shot Learning）能力是模型规模扩大的涌现特性。通过极大增加模型参数量和数据量，使模型具备上下文学习（In-Context Learning）能力，无需更新权重即可完成任务。
基于 GPT-3 架构，采用 Decoder-only 的 Transformer 结构。参数量达 1750 亿，训练数据涵盖多种领域。推理时提供任务描述和少量示例（Few-Shot），模型直接生成答案。
在多个 NLP 基准
- output: 可以这样理解《Language Models are Few-Shot Learners》：先看它要解决的任务，再看它为什么需要新的模型设计或训练策略。文档主线是 提出“少样本学习”（Few-Shot Learning）能力是模型规模扩大的涌现特性。通过极大增加模型参数量和数据量，使模型具备上下文学习（In-Context Learning）能力，无需更新权重即可完成任务。
基于 GPT-3 架构，采用 Decoder-only 的 Transformer 结构。参数量达 1750 亿，训练数据涵盖多种领域。推理时提供任务描述和少量示例（Few-Shot），模型直接生成答案。
在多个 NLP 基准测试中达到或接近 SOTA。特别是在

### train-5
- instruction: 请根据给定资料，输出稳定、简洁的学习卡片。
- input: 资料标题：LLaMA: Open and Efficient Foundation Language Models
来源文件：LLaMA Open and Efficient Foundation Language Models.md
内容摘录：专注于训练最优的基础语言模型，仅使用公开可用数据。通过架构优化和大规模训练，使较小参数量的模型达到甚至超越更大模型的性能，强调开放性与效率，验证公开数据训练大模型的可行性。
基于 Decoder
- output: 学习卡片
主题：LLaMA: Open and Efficient Foundation Language Models
关键词：LLaMA, Open, and, Efficient
核心内容：专注于训练最优的基础语言模型，仅使用公开可用数据。通过架构优化和大规模训练，使较小参数量的模型达到甚至超越更大模型的性能，强调开放性与效率，验证公开数据训练大模型的可行性。
基于 Decoder-only Transformer 架构。采用预归一化（RMSNorm）替代 LayerNorm 提升稳定性，使用 SwiGLU 激活函数增强表达能力，位置编码采用 Rotary Embeddings (RoPE)。训练数据仅包含公开集合，如 C
复

## val

### val-1
- instruction: 如果要向同学介绍《Training data-efficient image transformers & distillation through attention》，请给出一段 100 字以内的中文说明。
- input: 
- output: 《Training data-efficient image transformers & distillation through attention》是一篇围绕 Training 和相关模型/训练方法展开的学习资料，重点介绍 利用知识蒸馏（Knowledge Distillation），让强大的 CNN 教师模型指导 ViT 学生模型。引入特殊的"Distillation Token"，使模型能够同时学习真实标签和教师模型的软标签，弥补 Transformer 缺乏归纳偏置的缺陷。
基于 ViT 架构，在输入序列中增加一个可学习的蒸馏 Token。教师网络（如 ResNet）处理。适合快速理解该方向的核心问题、方法创新点和应用边

### val-2
- instruction: 请用 3 点概括《Attention Is All You Need》的核心思想。
- input: 资料标题：Attention Is All You Need
来源文件：Attention Is All You Need.md
内容摘录：抛弃循环（Recurrence）和卷积（Convolution），仅使用注意力机制（Attention Mechanism）来实现序列数据的建模。提出"Self-Attention"概念，让模型直接关注全局信息，实现任意位置间的直接交互。
采用标准的 Encoder-Decoder 架构。Encod
- output: 《Attention Is All You Need》的核心思想可以概括为：1）抛弃循环（Recurrence）和卷积（Convolution），仅使用注意力机制（Attention Mechanism）来实现序列数据的建模。提出"Self-Attention"概念，让模型直接关注全局信息，实现任意位置间的直接交互。
采用标准的 Encoder-Decoder 架构。Encoder 和 Decoder 均由 N 个相同层堆叠而成。每层包；2）关键方法线索包括 **作者**：Ashish Vaswani 等 (Google Brain / Google Research)；解决传统序列转导模型（如 RNN、LSTM、CNN）依赖串行计

### val-3
- instruction: 请提炼《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》中适合写入笔记的章节总结。
- input: 1. 基本信息: 作者：Colin Raffel 等 (Google Research)
年份：2019 (arXiv), 2020 (JMLR)
会议/期刊：Journal of Machine Learning Research (JMLR)
研究方向：自然语言处理 (NLP)、迁移学习、预训练模型
2. 这篇论文主要解决什么问题: 解决 NLP 任务碎片化问题，不同任务（分类、翻译、生成）需不同模型架构。旨在建立统一的文本到文本框
- output: 章节总结建议：围绕《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》，先记录背景与任务设定，再总结方法结构和训练要点，随后补充实验结论。根据当前资料，可优先记下：所有 NLP 任务均可转化为文本生成任务。输入为文本，输出也为文本。通过统一的 Transformer Encoder-Decoder 架构，实现多任务联合预训练与微调。
采用标准 Transformer Encoder-Decoder 结构。预训练任务为 Span Corruption（Span 掩码还原）。微调时，将任务类型作为前缀加入输入文本（如"tra

### val-4
- instruction: 请提炼《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》中适合写入笔记的章节总结。
- input: 1. 基本信息: 作者：Patrick Lewis 等 (Facebook AI Research)
年份：2020 (arXiv), 2021 (NeurIPS)
会议/期刊：NeurIPS 2020
研究方向：自然语言处理 (NLP)、检索增强生成、知识密集型任务
2. 这篇论文主要解决什么问题: 解决预训练语言模型知识存储有限、难以更新及容易产生幻觉（Hallucination）的问题。传统模型依赖参数记忆知识，无法访问外部最新信
- output: 章节总结建议：围绕《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》，先记录背景与任务设定，再总结方法结构和训练要点，随后补充实验结论。根据当前资料，可优先记下：将非参数化记忆（外部文档检索）与参数化记忆（模型权重）相结合。通过检索相关文档作为上下文，增强生成模型的知识能力，实现知识的高效利用与更新。
提出 RAG 框架，包含两个主要组件：检索器（Retriever）和生成器（Generator）。检索器通常基于 DPR（Dense Passage Retrieval）从外部知识库检索 Top-K 文档。生成器（如 BART 或 T5）将检索到的文档与

### val-5
- instruction: 请把《RoBERTa: A Robustly Optimized BERT Pretraining Approach》整理成一段简短教学式讲解。
- input: 重新评估并优化 BERT 的预训练超参数和策略。证明在相同架构下，通过更严谨的训练设计（如去除 NSP、动态掩码、更大批量）可以显著提升模型表现，强调工程优化的重要性。
架构与 BERT 完全一致（Transformer Encoder）。主要改进在于训练过程：移除下一句预测（NSP）任务，仅使用掩码语言模型（MLM）；采用动态掩码（Dynamic Masking），每次输入时重新生成掩码；增加训练步数、批量大小及数据量。
在 GLUE
- output: 可以这样理解《RoBERTa: A Robustly Optimized BERT Pretraining Approach》：先看它要解决的任务，再看它为什么需要新的模型设计或训练策略。文档主线是 重新评估并优化 BERT 的预训练超参数和策略。证明在相同架构下，通过更严谨的训练设计（如去除 NSP、动态掩码、更大批量）可以显著提升模型表现，强调工程优化的重要性。
架构与 BERT 完全一致（Transformer Encoder）。主要改进在于训练过程：移除下一句预测（NSP）任务，仅使用掩码语言模型（MLM）；采用动态掩码（Dynamic Masking），每次输入时重新生成掩码；增加训练步数、批量大小及数据量。
在 GLUE

## eval

### eval-1
- instruction: 请解释概念“LLaMA”，并说明它和《LLaMA: Open and Efficient Foundation Language Models》的关系。
- input: 资料标题：LLaMA: Open and Efficient Foundation Language Models
来源文件：LLaMA Open and Efficient Foundation Language Models.md
内容摘录：专注于训练最优的基础语言模型，仅使用公开可用数据。通过架构优化和大规模训练，使较小参数量的模型达到甚至超越更大模型的性能，强调开放性与效率，验证公开数据训练大模型的可行性。
基于 Decoder
- output: LLaMA 可以理解为这篇资料中的关键概念或方法线索。结合《LLaMA: Open and Efficient Foundation Language Models》，它通常用于说明模型结构、训练目标或知识迁移方式。学习时建议先记住它解决的问题，再对照方法部分理解实现思路。

### eval-2
- instruction: 请把《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》整理成一段简短教学式讲解。
- input: 将非参数化记忆（外部文档检索）与参数化记忆（模型权重）相结合。通过检索相关文档作为上下文，增强生成模型的知识能力，实现知识的高效利用与更新。
提出 RAG 框架，包含两个主要组件：检索器（Retriever）和生成器（Generator）。检索器通常基于 DPR（Dense Passage Retrieval）从外部知识库检索 Top-K 文档。生成器（如 BART 或 T5）将检索到的文档与输入问题拼接，生成最终答案。支持端到端微调。
- output: 可以这样理解《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》：先看它要解决的任务，再看它为什么需要新的模型设计或训练策略。文档主线是 将非参数化记忆（外部文档检索）与参数化记忆（模型权重）相结合。通过检索相关文档作为上下文，增强生成模型的知识能力，实现知识的高效利用与更新。
提出 RAG 框架，包含两个主要组件：检索器（Retriever）和生成器（Generator）。检索器通常基于 DPR（Dense Passage Retrieval）从外部知识库检索 Top-K 文档。生成器（如 BART 或 T5）将检索到的文档与输入问题拼接，生成最终答案

### eval-3
- instruction: 如果要向同学介绍《Language Models are Few-Shot Learners》，请给出一段 100 字以内的中文说明。
- input: 
- output: 《Language Models are Few-Shot Learners》是一篇围绕 Language 和相关模型/训练方法展开的学习资料，重点介绍 提出“少样本学习”（Few-Shot Learning）能力是模型规模扩大的涌现特性。通过极大增加模型参数量和数据量，使模型具备上下文学习（In-Context Learning）能力，无需更新权重即可完成任务。
基于 GPT-3 架构，采用 Decoder-only 的 Transformer 结构。参数量达 1750 亿，训练数据涵盖多种领域。推理时提供。适合快速理解该方向的核心问题、方法创新点和应用边界。

### eval-4
- instruction: 请基于《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》设计一个复习提问，并直接给出参考答案。
- input: 资料标题：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
来源文件：Megatron-LM Training Multi-Billion Parameter Language Models Using Model Parallelism.md
内容摘录：采用模型并行（Model Parallelism）策略，特别是
- output: 复习提问：阅读《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》时，最应该先抓住哪条主线？
参考答案：优先抓住“要解决的问题是什么、核心方法如何改进、这些改进带来什么收益和限制”这条主线，并结合 Megatron-LM 等关键词做归纳。

### eval-5
- instruction: 请解释概念“Training”，并说明它和《Training data-efficient image transformers & distillation through attention》的关系。
- input: 资料标题：Training data-efficient image transformers & distillation through attention
来源文件：Training data-efficient image transformers & distillation through attention.md
内容摘录：利用知识蒸馏（Knowledge Distillation），让强大的 CNN 教师模型指导 ViT
- output: Training 可以理解为这篇资料中的关键概念或方法线索。结合《Training data-efficient image transformers & distillation through attention》，它通常用于说明模型结构、训练目标或知识迁移方式。学习时建议先记住它解决的问题，再对照方法部分理解实现思路。

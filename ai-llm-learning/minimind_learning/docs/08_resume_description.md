# 简历描述

## 项目名称

MiniMind 轻量级 LLM 训练流程复现实验

## 简历项目描述

基于 MiniMind 复现轻量级 LLM 训练流程，在 AutoDL RTX 4090 环境下完成 CUDA/PyTorch 环境验证、Pretrain、SFT 和推理加载测试。整理 pretrain loss、权重文件、sha256 对比和服务器文件记录，验证训练过程能够生成新的模型参数。通过实验理解 tokenizer、Pretrain next-token prediction、SFT 指令微调和 eval 权重加载流程，并总结小参数模型在训练日志和知识问答方面的局限性。

## 1 分钟讲解稿

我做这个项目是为了把 LLM 底层训练流程实际跑一遍，而不是只停留在概念上。实验基于 MiniMind，在 AutoDL 的 RTX 4090 环境里先验证 PyTorch、CUDA 和 GPU 可用，然后跑 Pretrain 和 SFT。Pretrain 阶段我记录了 loss，从 7 左右下降到 2.x，并通过 sha256 对比确认 `pretrain_512.pth` 在训练后发生变化。SFT 阶段生成了 `full_sft_512.pth`，也通过 hash 和推理加载日志确认权重可用。这个项目的重点不是说小模型效果很好，而是我理解了数据、训练脚本、权重保存、hash 验证和推理加载这条完整链路。

## 面试可能追问

### 你为什么做 MiniMind？

- MiniMind 参数量较小，适合个人租用 GPU 跑完整训练流程。
- 可以直接接触 Pretrain、SFT 和 inference，而不只是调用 API。
- 适合作为理解 LLM 底层链路的学习项目。
- 目标是复现实验流程，不是训练生产级模型。

### Pretrain 和 SFT 的区别是什么？

- Pretrain 主要做 next-token prediction，让模型学习语言建模。
- SFT 使用指令或问答数据，让模型学习按照指令回答。
- Pretrain 更偏续写，SFT 更偏助手对话格式。
- 两者都可以看 loss，但最终评估重点不同。

### 你怎么证明模型真的重新训练了？

- Pretrain 前后 `pretrain_512.pth` 的 sha256 不同。
- SFT 中社区版和自己训练版 `full_sft_512.pth` 的 sha256 不同。
- `out_files.txt` 记录生成了新的权重文件。
- `sft_run.log` 记录 `eval_llm.py` 可以加载 `full_sft` 权重。

### 为什么 SFT 日志不完整还能确认训练有效？

- 日志不完整会影响 loss 分析，但不等于没有生成权重。
- 当前证据里有 `full_sft_512.pth` 文件记录。
- hash 对比显示自己训练权重与社区版权重不同。
- 推理加载日志显示模型参数可加载。
- 后续会用 `python -u` 和更小 `log_interval` 改进日志保存。

### 为什么模型回答质量不好？

- 小参数模型知识容量有限。
- 数据规模、训练轮数和专业知识覆盖有限。
- 本项目目标是训练流程复现，不是知识问答效果。
- 对专业知识问答，可以后续结合 RAG 或更高质量 SFT 数据。

### 这个项目和 RAG 项目有什么关系？

- MiniMind 项目帮助我理解模型训练和权重加载。
- RAG 项目更偏应用层，用检索增强模型回答。
- 小模型知识准确性有限时，RAG 可以补充外部知识。
- 两个项目结合起来，可以说明我既理解底层训练，也理解上层 LLM 应用。

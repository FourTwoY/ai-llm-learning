# Pretrain 与 SFT 对比

## 对比表

| 维度 | Pretrain | SFT |
| --- | --- | --- |
| 数据格式 | 普通文本或预训练语料，本项目记录为 `pretrain_hq.jsonl` | 指令/问答格式数据，具体 SFT 数据内容当前未展开记录 |
| 训练目标 | next-token prediction，学习语言建模能力 | 学习按照指令或对话格式回答 |
| labels 构造 | 通常让模型预测下一个 token | 通常只对需要学习回答的部分计算 loss，具体实现以 MiniMind 原代码为准 |
| 输出权重 | `out/pretrain_512.pth` | `out/full_sft_512.pth` |
| 评估方式 | loss 下降、权重 hash 变化、后续推理加载 | 权重生成、hash 对比、推理加载和问答观察 |
| 本项目实验结果 | loss 从约 7.08 下降到 2.x，最终记录 `2.218921`，权重 hash 变化 | 生成约 56M 权重，社区版和自己训练版 hash 不同，推理加载成功；问答对比显示常见问题相对通顺，专业概念错误较多 |

## 简洁解释

Pretrain 让模型通过 next-token prediction 学习语言建模能力，更偏向“续写”。SFT 使用指令数据，让模型学习按照用户问题给出更像助手的回答，更偏向“对话/助手格式”。

两者都可以通过 loss、权重文件和推理结果做验证，但评价重点不同：Pretrain 更关注语言建模训练是否正常，SFT 更关注模型是否能按指令格式输出。

## 面试回答版本

这个项目里我主要复现了 MiniMind 的轻量级训练流程。Pretrain 阶段用 `pretrain_hq.jsonl` 做 next-token prediction，观察 loss 从 7 左右下降到 2.x，并通过 sha256 对比确认训练后的 `pretrain_512.pth` 发生了变化。SFT 阶段生成了 `full_sft_512.pth`，并和社区版权重做 hash 对比，确认自己训练出的权重文件不同。这个实验的重点不是让小模型回答得很好，而是把环境检查、数据读取、训练、权重保存和推理加载这条链路跑通，并理解 Pretrain 和 SFT 在目标上的区别。

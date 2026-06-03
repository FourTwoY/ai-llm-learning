# MiniMind 轻量级 LLM 训练流程复现实验

本项目用于记录基于 MiniMind 的轻量级 LLM 底层训练流程复现。实验覆盖环境验证、Pretrain、SFT、权重验证和推理加载验证，目标不是追求模型效果，而是理解 LLM 从数据、训练到推理的完整链路。

项目面向 AI/LLM 实习项目展示，文档内容只整理已经放入 `tmp_minimind_records/` 的服务器实验记录。数据集、模型权重、checkpoint 和大日志不提交到 GitHub。

## 已完成内容

- AutoDL RTX 4090 环境验证
- PyTorch / CUDA / GPU 可用性检查
- MiniMind Pretrain 实验
- MiniMind SFT 实验
- `pretrain_512.pth` 与 `full_sft_512.pth` 权重生成记录
- `sha256sum` 对比验证训练前后/不同来源权重发生变化
- `eval_llm.py --weight full_sft` 推理加载验证
- 社区权重与自己训练权重的问答对比摘要
- SFT 日志不完整问题复盘

说明：推理问答原文来自本地补充记录 `minimind.md`，文档中只保留摘要，不复制长回答和本地图片路径。

## 核心实验结果摘要

| 实验项目 | 输入数据 | 输出文件 | 验证方式 | 观察结论 |
| --- | --- | --- | --- | --- |
| 环境检查 | AutoDL 服务器环境 | `env_check.txt`、`nvidia_smi*.txt` | PyTorch/CUDA/GPU 信息 | PyTorch 为 `2.6.0+cu124`，torch CUDA 为 `12.4`，GPU 为 RTX 4090，CUDA 可用 |
| Pretrain | `pretrain_hq.jsonl` | `out/pretrain_512.pth` | loss 日志、文件大小、sha256 对比 | loss 从约 7.08 降到约 2.x，训练前后 hash 不同，说明权重发生变化 |
| SFT | mini SFT 数据集 | `out/full_sft_512.pth` | 文件列表、sha256 对比、推理加载日志 | 生成约 56M 权重，社区版与自己训练版 hash 不同，`eval_llm.py` 可加载模型 |
| 推理对比 | `full_sft_512.pth` | 终端交互输出 | `minimind.md`、`sft_run.log` | 两个权重都能生成中文回答；常见问题回答较通顺，专业概念存在明显误解 |

## 项目目录

```text
minimind_learning/
├── README.md
├── docs/
│   ├── 01_project_overview.md
│   ├── 02_experiment_environment.md
│   ├── 03_inference_evaluation.md
│   ├── 04_pretrain_experiment.md
│   ├── 05_sft_experiment.md
│   ├── 06_pretrain_vs_sft.md
│   ├── 07_problems_and_solutions.md
│   └── 08_resume_description.md
├── experiments/
│   ├── exp01_environment_check.md
│   ├── exp02_pretrain_run.md
│   ├── exp03_sft_run.md
│   └── exp04_inference_comparison.md
├── evidence/
│   ├── README.md
│   ├── pretrain_loss_summary.md
│   ├── sha256_summary.md
│   └── server_file_summary.md
└── prompts/
    └── codex_project_polish_prompt.md
```

## 学到的内容

- Pretrain 的核心是 next-token prediction，让模型学习语言建模能力。
- SFT 使用指令数据，让模型更接近对话助手的回答格式。
- 小参数模型可以跑通完整训练链路，但知识准确性和专业概念理解能力有限。
- 日志记录需要注意 Python 输出缓冲、`tee` 管道和 `log_interval`，否则可能出现训练完成但日志不完整的情况。

## 注意事项

- 数据集和权重文件没有提交到 GitHub。
- 截图需要后续手动放入 `screenshots/` 或 GitHub README 图片位置。
- 本项目是学习复现实验，不是生产级模型训练项目。
- 当前截图仍在本地 Typora 图片目录，后续建议统一复制到项目截图目录或使用 GitHub 图片链接。

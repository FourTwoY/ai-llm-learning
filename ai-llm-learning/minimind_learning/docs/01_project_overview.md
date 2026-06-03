# 项目概览

## 项目定位

本项目记录基于 MiniMind 的轻量级 LLM 训练流程复现实验。实验运行在 AutoDL 租用服务器上，重点覆盖环境验证、Pretrain、SFT、权重文件验证和推理加载验证。

这个项目不把模型回答效果作为主要目标，而是把重点放在训练链路本身：数据文件如何被训练脚本读取，训练过程如何产生 loss 日志，权重文件如何保存，如何通过 hash 和推理加载确认训练结果。

## 已整理的实验记录

| 模块 | 记录来源 | 当前状态 |
| --- | --- | --- |
| 环境检查 | `env_check.txt`、`nvidia_smi.txt`、`nvidia_smi_after.txt` | 已整理 |
| 数据集与服务器文件 | `dataset_files.txt`、`out_files.txt`、`git_status.txt` | 已整理 |
| Pretrain | `pretrain_loss_lines.txt`、`pretrain_log_tail.txt`、`pretrain_*sha256.txt`、`pretrain_weight_info.txt` | 已整理 |
| SFT | `sft_run.log`、`out_files.txt`、`sha256_compare.txt` | 已整理 |
| 推理问答对比 | 本地补充记录 `minimind.md` | 已整理摘要 |

## 实验边界

- 不修改 MiniMind 原项目源码。
- 不把数据集、权重、checkpoint、out 目录、大日志提交到 GitHub。
- 不夸大模型效果。
- 缺失记录统一写“未记录”或“后续补充”。

## 适合展示的重点

1. 能说明自己实际跑过 LLM 训练流程，而不是只读理论。
2. 能用 loss、文件大小、hash 和推理加载记录解释“训练是否真的发生”。
3. 能主动说明小模型回答质量有限，体现对实验边界的理解。
4. 能总结日志不完整、训练产物过大等工程问题。

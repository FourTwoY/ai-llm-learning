# 实验环境

## 环境摘要

| 项目 | 记录值 |
| --- | --- |
| 服务器平台 | AutoDL |
| GPU | NVIDIA GeForce RTX 4090 |
| 显存 | 约 24GB，`nvidia-smi` 记录为 `24564MiB` |
| PyTorch | `2.6.0+cu124` |
| torch CUDA | `12.4` |
| nvidia-smi CUDA Version | `12.6` |
| `torch.cuda.is_available()` | `True` |

## 记录来源

环境信息来自以下文件：

- `tmp_minimind_records/minimind_pretrain_records/experiment_records/env_check.txt`
- `tmp_minimind_records/minimind_pretrain_records/experiment_records/nvidia_smi.txt`
- `tmp_minimind_records/minimind_pretrain_records/experiment_records/nvidia_smi_after.txt`
- `tmp_minimind_records/minimind_experiment_records/experiment_records/env_check.txt`

## nvidia-smi 检查

训练前记录时间为 2026-06-02 21:52:03，`nvidia-smi` 显示：

- Driver Version: `560.35.03`
- CUDA Version: `12.6`
- GPU: `NVIDIA GeForce RTX 4090`
- Memory: `1MiB / 24564MiB`
- 无运行中的 GPU 进程

训练后记录时间为 2026-06-02 22:33:10，`nvidia-smi` 仍显示 GPU 可用：

- GPU: `NVIDIA GeForce RTX 4090`
- Memory: `1MiB / 24564MiB`
- 无运行中的 GPU 进程

## 关于 CUDA 版本

记录中 torch 显示的 CUDA 版本是 `12.4`，`nvidia-smi` 显示的 CUDA Version 是 `12.6`。这两个值不完全一致是常见情况：前者表示 PyTorch 构建/运行时使用的 CUDA 版本，后者表示驱动支持的 CUDA 版本上限。

本次记录只说明 CUDA 与 GPU 可用，不把这种版本差异写成错误。

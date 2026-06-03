# 实验 01：环境检查

## 目的

确认 AutoDL 服务器上的 PyTorch、CUDA 和 GPU 是否可用于 MiniMind 训练。

## 检查命令记录

原始命令未完整记录，当前结果来自：

- `env_check.txt`
- `nvidia_smi.txt`
- `nvidia_smi_after.txt`

## 检查结果

| 检查项 | 结果 |
| --- | --- |
| PyTorch | `2.6.0+cu124` |
| torch CUDA | `12.4` |
| `torch.cuda.is_available()` | `True` |
| GPU | `NVIDIA GeForce RTX 4090` |
| nvidia-smi CUDA Version | `12.6` |
| 显存 | `24564MiB`，约 24GB |

## 训练前后 GPU 状态

训练前后 `nvidia-smi` 均能识别 RTX 4090，并显示 GPU 可用。记录中没有运行中的 GPU 进程，说明检查时 GPU 处于空闲或已释放状态。

## 结论

AutoDL RTX 4090 环境满足本次 MiniMind Pretrain 和 SFT 复现实验的基本运行条件。

torch CUDA 版本与 `nvidia-smi` CUDA Version 不完全相同，这属于常见情况，不作为环境错误处理。

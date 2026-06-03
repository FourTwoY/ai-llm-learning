# 实验 03：SFT 运行记录

## 实验目标

运行 MiniMind SFT 流程，生成 `full_sft_512.pth`，并通过 hash 与推理加载验证权重文件。

## 运行结果

根据 `out_files.txt`：

```text
total 112M
full_sft_512.pth  约 56M
pretrain_512.pth  约 56M
```

根据 `sft_run.log`：

```text
所加载Model可训练参数：25.830 百万
```

## sha256 对比

| 权重 | sha256 |
| --- | --- |
| 社区版 full_sft_512.pth | `21d4c07c9d5029047b3cdefc653ed68c678ba1e95e193b8f176459edfff63f78` |
| 自己训练 full_sft_512.pth | `1bae82f1c33a86805b3097d556ce96dca6c8e8e4e8de43be4bd2e6ae3f6e0334` |

## 日志问题

`sft_run.log` 没有完整 step/loss。当前只能确认模型参数加载和权重 hash 差异，不能整理完整 SFT loss 曲线。

后续建议：

- 使用 `python -u train_full_sft.py ...`
- 使用 `tee` 保存完整日志
- 缩短 `log_interval`
- 避免手动 kill 或重复启动训练

## 结论

SFT 阶段生成了 `full_sft_512.pth`，且自己训练版本与社区版本 hash 不同。结合推理加载日志，可以确认 SFT 产物可被加载，但完整训练 loss 需要后续补记录。

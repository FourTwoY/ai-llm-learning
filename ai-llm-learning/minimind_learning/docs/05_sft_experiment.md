# SFT 实验记录

## 运行结果

根据 `out_files.txt` 和 `sft_run.log`：

- 已运行 `train_full_sft.py`。
- `out` 目录中生成了 `full_sft_512.pth`。
- `full_sft_512.pth` 文件大小约 `56M`。
- `sft_run.log` 显示推理加载时模型可训练参数约为 `25.830` 百万。

## 权重验证

根据 `sha256_compare.txt`，两个 `full_sft_512.pth` 的 hash 不同：

| 权重来源 | sha256 |
| --- | --- |
| 社区版 full_sft_512.pth | `21d4c07c9d5029047b3cdefc653ed68c678ba1e95e193b8f176459edfff63f78` |
| 自己训练生成 full_sft_512.pth | `1bae82f1c33a86805b3097d556ce96dca6c8e8e4e8de43be4bd2e6ae3f6e0334` |

hash 不同，说明自己训练后产生了一个不同的权重文件。但这不代表效果更好，只能说明文件内容发生了变化。

## 推理验证

根据 `sft_run.log` 和本地补充记录 `minimind.md`：

- `eval_llm.py --weight full_sft` 可以加载模型。
- 日志中出现“所加载Model可训练参数：25.830 百万”。
- 可以进入手动输入模式。
- 可以回答中文问题。

## SFT 日志不完整

本次 `sft_run.log` 只记录了环境 warning 和模型参数加载信息，没有完整 step/loss。

可能原因：

- `log_interval` 较大。
- `tee` 管道或 stdout 存在输出缓冲。
- 训练进程曾被重复启动或手动 kill，导致 stdout 未完整 flush。

虽然日志不完整，但根据当前记录，仍可通过以下方式确认 SFT 产生了新权重：

- `out_files.txt` 中存在 `full_sft_512.pth`。
- `sha256_compare.txt` 显示自己训练权重与社区版权重 hash 不同。
- `sft_run.log` 显示 `eval_llm.py` 可加载 `full_sft` 权重。

后续复现实验建议：

- 使用 `python -u` 运行训练脚本。
- 使用 `tee` 保存日志。
- 缩短 `log_interval`。
- 避免重复启动训练和手动 kill。
- 训练结束后检查权重时间、hash 和推理结果。

## 模型效果观察

根据 `minimind.md` 记录：

- 模型能完成基本对话格式。
- 对“机器学习”这类常见问题能给出较通顺回答。
- 对“监督微调”“Pretrain 和 SFT 区别”“RAG”等专业概念存在明显误解。
- 说明小参数模型在知识准确性和概念理解方面有限。
- 本实验主要验证训练与推理流程，而不是追求回答质量。

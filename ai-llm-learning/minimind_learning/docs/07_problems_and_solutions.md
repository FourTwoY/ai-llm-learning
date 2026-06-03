# 问题与解决方案

## 问题 1：SFT 日志不完整

### 现象

`sft_run.log` 只记录了环境 warning 和模型参数加载信息，没有完整 step/loss。

### 可能原因

- `log_interval` 较大，短日志中没有捕获到训练 step。
- `tee` 管道或 Python stdout 存在输出缓冲。
- 训练进程可能被重复启动或手动 kill，导致 stdout 未完整 flush。

### 本次如何验证训练有效

- `out_files.txt` 记录 `full_sft_512.pth` 已生成，大小约 `56M`。
- `sha256_compare.txt` 记录社区版权重和自己训练权重 hash 不同。
- `sft_run.log` 记录 `eval_llm.py --weight full_sft` 可加载模型，参数约 `25.830` 百万。

### 后续改进

- 使用 `python -u` 关闭输出缓冲。
- 使用 `tee` 保存日志。
- 缩短 `log_interval`。
- 避免重复启动和手动 kill。
- 训练结束后检查权重时间、hash 和推理结果。

## 问题 2：训练产生大量 untracked 文件

### 现象

`git_status.txt` 显示 MiniMind 原仓库有大量未跟踪文件，包括：

- `checkpoints/`
- `dataset/pretrain_hq.jsonl`
- `dataset/sft_512.jsonl`
- `dataset/sft_mini_512.jsonl`
- `out_backup/`
- `__pycache__/`
- `.ipynb_checkpoints/`

### 原因

训练数据、checkpoint、权重输出和缓存文件都属于训练产物，不应该直接提交到 GitHub。

### 处理方式

- 保留在服务器或本地磁盘。
- GitHub 只提交文档、摘要和小日志。
- 使用 `.gitignore` 忽略大文件和训练产物。

## 问题 3：小模型回答质量有限

### 现象

本地补充记录 `minimind.md` 中的问答对比显示，模型对“机器学习”这类常见问题能生成较通顺回答，但对“监督微调”“Pretrain 和 SFT 区别”“RAG”等专业概念存在明显误解和偏题。

### 可能原因

- 模型参数量较小。
- 数据规模和数据质量有限。
- 训练轮数有限。
- 专业知识覆盖不足。

### 处理方式

- 不夸大效果。
- 在 README 中定位为训练流程复现实验。
- 后续如果要改善知识准确性，可以结合 RAG 或更高质量 SFT 数据。

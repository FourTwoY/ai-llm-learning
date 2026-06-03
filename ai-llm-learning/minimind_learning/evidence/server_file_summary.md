# 服务器文件摘要

## dataset 目录

记录来源：`dataset_files.txt`

| 文件/目录 | 大小 | 说明 |
| --- | --- | --- |
| `__init__.py` | 0 | Python 包标记文件 |
| `__pycache__/` | 84B | Python 缓存目录 |
| `dataset.md` | 141B | 数据集说明文件 |
| `lm_dataset.py` | 9.2K | 数据集读取相关代码 |
| `pretrain_hq.jsonl` | 1.6G | Pretrain 数据 |
| `sft_512.jsonl` | 7.1G | SFT 数据 |
| `sft_mini_512.jsonl` | 1.2G | mini SFT 数据 |

`dataset` 总大小约 `9.8G`，不适合提交到 GitHub。

## out 目录

记录来源：`out_files.txt`

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `full_sft_512.pth` | 约 56M | SFT 后生成的权重 |
| `pretrain_512.pth` | 约 56M | Pretrain 后生成的权重 |

权重文件不提交到 GitHub。

## MiniMind 原仓库状态

记录来源：`git_status.txt`

- MiniMind 原仓库处于 `master` 分支。
- 当前分支与 `origin/master` 同步。
- 训练产生了大量 untracked 文件，包括 `checkpoints/`、`dataset/*.jsonl`、`out_backup/`、`__pycache__/`、`.ipynb_checkpoints/` 等。
- 这些文件是训练产物或缓存，不应该提交到 GitHub。

## 处理原则

- 保留原始文件在服务器或本地。
- GitHub 只提交整理后的文档和证据摘要。
- 通过 `.gitignore` 忽略数据集、权重、checkpoint、日志和缓存。

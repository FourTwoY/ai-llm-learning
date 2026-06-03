# Codex 项目整理提示词

你现在是我的 GitHub 项目整理助手。请基于我已经完成的 MiniMind Pretrain 和 SFT 实验记录，完善 `ai-llm-learning` 仓库中的 `minimind_learning` 项目；如果没有该目录，则创建。

重要原则：

1. 不要修改 MiniMind 原项目源码。
2. 不要提交模型权重、数据集、大日志、checkpoint、out 目录。
3. 不要夸大实验效果。
4. 重点是把已经完成的服务器实验整理成 GitHub 可展示文档。
5. 所有结论必须来自 `tmp_minimind_records/` 里的记录，不要虚构。
6. 如果某项信息记录里没有，就写“未记录”或“后续补充”。

需要完成：

- 检查并完善 `.gitignore`，忽略 `tmp_minimind_records/`、`checkpoints/`、`out/`、`out_backup/`、`dataset/`、`*.pth`、`*.bin`、`*.safetensors`、`wandb/`、`runs/`、`logs/`、`__pycache__/`、`.ipynb_checkpoints/`、`.pytest_cache/`。
- 创建或补充 `minimind_learning/README.md`，说明项目定位、已完成内容、核心实验结果、目录说明、学到的内容和注意事项。
- 创建 `docs/`、`experiments/`、`evidence/`、`prompts/` 目录，并按实验记录整理文档。
- 从 `env_check.txt`、`nvidia_smi.txt`、`nvidia_smi_after.txt` 整理 AutoDL RTX 4090、PyTorch、CUDA 和 GPU 可用性。
- 从 `dataset_files.txt`、`out_files.txt`、`git_status.txt` 整理服务器文件、数据集、权重和不应提交的训练产物。
- 从 `pretrain_script_data_path.txt`、`pretrain_loss_lines.txt`、`pretrain_log_tail.txt`、`pretrain_before_sha256.txt`、`pretrain_after_sha256.txt`、`pretrain_weight_info.txt` 整理 Pretrain 实验。
- 从 `sft_run.log`、`out_files.txt`、`sha256_compare.txt` 整理 SFT 实验。
- 如果存在 `minimind.md`，从中整理社区版权重和自己训练权重的问答对比；如果不存在，标注“未记录/后续补充”。
- 整理 Pretrain vs SFT 对比、问题与解决方案、简历描述和面试追问。
- 最后运行 `git status`，检查 `tmp_minimind_records/`、`dataset/`、`out/`、`checkpoints/`、`*.pth` 没有进入待提交。

推荐 commit message：

```text
docs: add minimind pretrain and sft experiment records
```

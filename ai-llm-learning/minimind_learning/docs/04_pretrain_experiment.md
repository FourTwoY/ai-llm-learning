# Pretrain 实验记录

## 入口脚本与数据路径

根据 `pretrain_script_data_path.txt`，`train_pretrain.py` 中记录到以下关键信息：

- 引入 `PretrainDataset`
- 默认 `data_path` 为 `../dataset/pretrain_hq.jsonl`
- 使用 `PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)` 构造训练数据

本次 Pretrain 使用的数据文件为 `pretrain_hq.jsonl`。

## 训练过程

根据 `pretrain_loss_lines.txt` 和 `pretrain_log_tail.txt`：

- 总 step 约为 `44160`
- 训练早期 loss 从 `7.082532` 开始下降
- 中后期 loss 基本在 `2.x` 范围波动
- 最后记录到 `44159/44160`，loss 为 `2.218921`
- 学习率从约 `0.0005` 逐步下降到约 `0.00005`

关键 loss 摘要见 [pretrain_loss_summary.md](../evidence/pretrain_loss_summary.md)。

## 输出权重

根据 `pretrain_weight_info.txt`：

```text
out/pretrain_512.pth  约 56M
```

## sha256 验证

训练前 hash：

```text
ec8b5c43191f714a8956de6d7dcb89ba18db12164925ad8107a876d4d48341c6
```

训练后 hash：

```text
ca2b2592916190d8fc641823a612ddbfc677e47f8e12ade8f7479aa1a489d837
```

两次 hash 不同，说明 `out/pretrain_512.pth` 在训练后确实发生了变化。

## 实验观察

- loss 整体下降明显，说明 Pretrain 训练流程正常运行。
- 后期 loss 存在波动，这在小参数模型和 mini/单轮训练中比较常见。
- 本实验主要验证 Pretrain 流程跑通，不评价最终模型的知识能力。

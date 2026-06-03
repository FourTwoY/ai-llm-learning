# 实验 02：Pretrain 运行记录

## 实验目标

运行 MiniMind Pretrain 流程，观察 loss 是否下降，并验证训练后 `pretrain_512.pth` 权重文件是否发生变化。

## 数据与脚本

| 项目 | 记录 |
| --- | --- |
| 入口脚本 | `train_pretrain.py` |
| Dataset 类 | `PretrainDataset` |
| 默认数据路径 | `../dataset/pretrain_hq.jsonl` |
| 实际训练数据 | `pretrain_hq.jsonl` |
| Dataset 构造 | `PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)` |

## 训练日志摘要

| step | loss | lr |
| --- | --- | --- |
| 100 | 7.082532 | 0.000499994306 |
| 1000 | 5.223985 | 0.000499430871 |
| 5000 | 2.820743 | 0.000485915217 |
| 10000 | 2.546142 | 0.000445424256 |
| 20000 | 2.237921 | 0.000308172683 |
| 30000 | 2.154793 | 0.000154828454 |
| 40000 | 2.246559 | 0.000059781573 |
| 44159 | 2.218921 | 0.000050000001 |

## 权重输出

根据 `pretrain_weight_info.txt`：

```text
out/pretrain_512.pth  约 56M
```

## sha256 对比

| 阶段 | sha256 |
| --- | --- |
| 训练前 | `ec8b5c43191f714a8956de6d7dcb89ba18db12164925ad8107a876d4d48341c6` |
| 训练后 | `ca2b2592916190d8fc641823a612ddbfc677e47f8e12ade8f7479aa1a489d837` |

## 结论

Pretrain loss 从 7.x 下降到 2.x，训练后权重 hash 与训练前不同。本次实验可以说明 Pretrain 流程正常运行，并生成了发生变化的 `pretrain_512.pth` 权重。

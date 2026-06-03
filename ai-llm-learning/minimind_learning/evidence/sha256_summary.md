# sha256 摘要

## Pretrain 权重

| 阶段 | 文件 | sha256 |
| --- | --- | --- |
| 训练前 | `out/pretrain_512.pth` | `ec8b5c43191f714a8956de6d7dcb89ba18db12164925ad8107a876d4d48341c6` |
| 训练后 | `out/pretrain_512.pth` | `ca2b2592916190d8fc641823a612ddbfc677e47f8e12ade8f7479aa1a489d837` |

结论：Pretrain 前后 hash 不同，说明权重文件发生变化。

## SFT 权重

| 权重来源 | 文件 | sha256 |
| --- | --- | --- |
| 社区版 | `full_sft_512.pth` | `21d4c07c9d5029047b3cdefc653ed68c678ba1e95e193b8f176459edfff63f78` |
| 自己训练版 | `full_sft_512.pth` | `1bae82f1c33a86805b3097d556ce96dca6c8e8e4e8de43be4bd2e6ae3f6e0334` |

结论：两个 SFT 权重 hash 不同，说明文件内容不同。hash 不同不能直接代表回答质量更好。

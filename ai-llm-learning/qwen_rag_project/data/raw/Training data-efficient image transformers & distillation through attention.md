# 论文标题：Training data-efficient image transformers & distillation through attention

## 1. 基本信息
- 作者：Hugo Touvron 等 (Facebook AI Research)
- 年份：2020 (arXiv), 2021 (ICML)
- 会议/期刊：ICML 2021
- 研究方向：计算机视觉 (CV)、模型蒸馏、Transformer

## 2. 这篇论文主要解决什么问题
解决 Vision Transformer (ViT) 依赖大规模数据集（如 JFT-300M）才能超越 CNN 的问题。在标准数据集（如 ImageNet-1K）上，原始 ViT 容易过拟合且性能不如 ResNet。旨在通过知识蒸馏技术，使 Transformer 在小数据上也能高效训练并取得优异性能。

## 3. 核心思想
利用知识蒸馏（Knowledge Distillation），让强大的 CNN 教师模型指导 ViT 学生模型。引入特殊的"Distillation Token"，使模型能够同时学习真实标签和教师模型的软标签，弥补 Transformer 缺乏归纳偏置的缺陷。

## 4. 模型/方法概述
基于 ViT 架构，在输入序列中增加一个可学习的蒸馏 Token。教师网络（如 ResNet）处理相同图像生成软目标。学生网络（DeiT）通过分类 Token 预测真实标签，通过蒸馏 Token 匹配教师输出。损失函数为交叉熵与蒸馏损失的加权和。

## 5. 关键创新点
- **蒸馏 Token (Distillation Token)**：专为蒸馏设计的独立 Token，避免干扰分类 Token 的学习，有效传递教师知识。
- **数据高效性**：无需外部大规模私有数据，仅在 ImageNet-1K 上训练即可达到 SOTA，降低了训练门槛。
- **CNN 教 Transformer**：证明了具有归纳偏置的 CNN 可以有效引导无偏置的 Transformer，加速收敛。

## 6. 实验结论
在 ImageNet 分类任务上，DeiT 性能显著优于原始 ViT，并与高效 CNN 模型持平或超越。证明了在有限数据下，蒸馏策略能有效提升 Transformer 的泛化能力和训练稳定性。

## 7. 这篇论文和其他论文的关系
- 和 Transformer 的关系：优化了 Transformer 在视觉领域的训练策略，增强了其数据适应性。
- 和 BERT / GPT / ViT / MAE 的关系：
    - **BERT**：蒸馏思路类似 DistilBERT，但 DeiT 针对视觉架构设计了专用 Token。
    - **GPT**：同为 Transformer 架构，但 GPT 侧重生成，DeiT 侧重分类理解。
    - **ViT**：本文即 DeiT，是 ViT 的直接改进版，解决了 ViT 数据饥渴问题。
    - **MAE**：MAE 通过自监督掩码解决数据问题，DeiT 通过 supervised 蒸馏解决，路径不同但目标一致。
    - **Megatron**：DeiT 模型若扩大规模，也可利用 Megatron 进行并行加速。

## 8. 我的理解
DeiT 是 ViT 走向实用的关键一步。它揭示了 Transformer 在视觉上的短板可通过蒸馏弥补，使普通研究者无需海量数据也能使用 ViT。蒸馏 Token 的设计巧妙解耦了任务目标与知识迁移。虽然后期自监督（如 MAE）成为主流，但 DeiT 的蒸馏思想在模型压缩和特定任务迁移中仍具重要价值。

## 9. 关键词
DeiT
ViT
Distillation
Data-efficient
Computer Vision
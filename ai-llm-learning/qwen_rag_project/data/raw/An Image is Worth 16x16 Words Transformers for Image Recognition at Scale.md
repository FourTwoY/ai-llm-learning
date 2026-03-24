# 论文标题：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

### 1. 基本信息
- **作者**：Alexey Dosovitskiy 等 (Google Research)
- **年份**：2020 (arXiv), 2021 (ICLR)
- **会议/期刊**：ICLR 2021
- **研究方向**：计算机视觉 (CV)、深度学习、图像分类、架构迁移

### 2. 这篇论文主要解决什么问题
解决卷积神经网络（CNN）在视觉任务中主导但存在归纳偏置（如局部性、平移等价性）限制的问题。探索是否可以直接将纯 Transformer 架构应用于图像识别，减少人工设计的归纳偏置，依赖大规模数据驱动学习全局特征。

### 3. 核心思想
将图像视为序列数据。把图像分割为固定大小的 Patch，线性嵌入为向量序列，直接输入标准 Transformer Encoder 进行训练，证明在充足数据下，纯注意力机制可超越 CNN。

### 4. 模型/方法概述
输入图像被分割为固定大小的 Patch（如 16x16）。每个 Patch 经线性投影得到 Embedding，加上位置编码（Position Embedding）和可学习的 Class Token。序列送入标准 Transformer Encoder，最后通过 MLP Head 对 Class Token 进行分类。

### 5. 关键创新点
- **纯 Transformer 架构**：在图像识别中完全移除卷积操作，仅依赖自注意力机制捕捉全局依赖。
- **Patch Embedding**：将二维图像转化为一维序列，建立了 NLP 与 CV 之间的通用建模范式。
- **数据驱动的 Scaling**：发现模型性能高度依赖预训练数据量，在大规模数据集（如 JFT-300M）上预训练后效果显著优于 CNN。

### 6. 实验结论
在 ImageNet、CIFAR-100 等多个基准测试上达到 State-of-the-Art (SOTA)。特别是在大规模预训练后，ViT-L/16 模型在 ImageNet 上准确率超越当时的最佳 CNN 模型，验证了架构的可扩展性。

### 7. 这篇论文和其他论文的关系
- **和 Transformer 的关系**：将 Transformer 架构成功迁移至计算机视觉领域，证明了其跨模态通用性。
- **和 BERT / GPT / ViT / MAE / Megatron 的关系**：
    - **BERT**：结构类似（均为 Encoder），但 BERT 处理文本序列，ViT 处理图像 Patch 序列。
    - **GPT**：GPT 为 Decoder 架构用于生成，ViT 为 Encoder 架构用于分类理解。
    - **ViT**：本文即 ViT 原论文，确立了视觉 Transformer 的标准范式。
    - **MAE**：基于 ViT 架构，改进预训练任务为掩码自编码，提升自监督学习效果。
    - **Megatron**：为 ViT 等大规模视觉模型提供并行训练支持，实现模型 Scaling。

### 8. 我的理解
ViT 是计算机视觉领域的里程碑，标志着 CV 从 CNN 时代进入 Transformer 时代。它揭示了“归纳偏置”与“数据规模”的权衡关系：当数据足够大时，弱归纳偏置的模型更具上限。这为后续多模态大模型（如 CLIP）的统一架构奠定了基础。

### 9. 关键词
Vision Transformer (ViT); Image Classification; Patch Embedding; Self-Attention; Transfer Learning
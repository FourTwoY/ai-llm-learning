# 论文标题：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 1. 基本信息
- 作者：Ze Liu 等 (Microsoft Research Asia)
- 年份：2021 (arXiv), 2021 (ICCV)
- 会议/期刊：ICCV 2021 (Best Paper)
- 研究方向：计算机视觉 (CV)、深度学习、骨干网络

## 2. 这篇论文主要解决什么问题
解决 Vision Transformer (ViT) 缺乏层次化结构导致难以适应密集预测任务（如检测、分割）的问题。ViT 固定分辨率且计算复杂度随图像尺寸平方增长，限制了其在多尺度视觉任务中的应用及高分辨率图像处理。

## 3. 核心思想
构建层次化 Transformer 架构。借鉴 CNN 的金字塔结构，通过 Patch Merging 逐步降低分辨率。引入移位窗口（Shifted Window）机制，在局部计算注意力效率的同时实现跨窗口连接，兼顾局部性与全局性。

## 4. 模型/方法概述
模型分为多个 Stage，每个 Stage 包含 Patch Merging 层和 Swin Transformer Blocks。Block 内部交替使用基于窗口的多头自注意力（W-MSA）和移位窗口多头自注意力（SW-MSA）。前者限制在局部窗口内计算，后者通过移位实现窗口间信息交互，无需额外参数。

## 5. 关键创新点
- **层次化架构**：生成多尺度特征图，便于下游任务如目标检测和分割的接入，类似 CNN 的 Feature Pyramid。
- **移位窗口机制**：在不显著增加计算量的前提下，打破窗口隔离，建立全局依赖，增强模型表达能力。
- **线性计算复杂度**：注意力计算限制在窗口内，复杂度随图像尺寸线性增长，而非平方级，支持高分辨率输入。

## 6. 实验结论
在 ImageNet 分类、COCO 目标检测、ADE20K 语义分割上均达到 State-of-the-Art (SOTA)。获得 ICCV 2021 最佳论文，证明了 Transformer 可作为通用视觉骨干网络，性能超越高效 CNN。

## 7. 这篇论文和其他论文的关系
- 和 Transformer 的关系：基于 Transformer 架构，针对视觉特性进行了结构性改良，是 Transformer 在 CV 领域的深度定制。
- 和 BERT / GPT / ViT / MAE 的关系：
    - **BERT**：均为 Transformer 变体，但 BERT 处理文本，Swin 处理视觉层次。
    - **GPT**：GPT 为自回归生成，Swin 为层次化编码理解。
    - **ViT**：Swin 是 ViT 的改进版，解决了 ViT 难以处理高分辨率和密集预测的问题。
    - **MAE**：Swin 常作为 MAE 的骨干网络，二者结合可进一步提升自监督学习效果。
    - **Megatron**：基于分层窗口注意力机制，可在 Megatron-LM 分布式训练框架下扩展为大规模视觉 - 语言多模态模型。

## 8. 我的理解
Swin Transformer 是视觉 backbone 的里程碑，真正实现了 Transformer 在 CV 领域的通用化。它巧妙平衡了局部性与全局性，既保留了 CNN 的层次化优势，又拥有 Attention 的动态建模能力。虽然后期出现了更多高效架构，但 Swin 提出的移位窗口思想影响深远，是理解现代视觉 Transformer 的关键，也是连接 CNN 与 Transformer 架构的重要桥梁。

## 9. 关键词
Swin Transformer
Hierarchical
Shifted Window
Computer Vision
Backbone
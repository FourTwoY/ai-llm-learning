# qwen_rag_project

一个基于 **FastAPI + 阿里云百炼（千问）** 的简易 RAG（Retrieval-Augmented Generation，检索增强生成）项目。

本项目从原始文档读取开始，完成了文档清洗、文本切分、向量化、相似度检索、rerank 精排、答案生成，并最终封装成可调用的 FastAPI 接口，适合作为一个可展示、可扩展的 RAG 后端入门项目。

---

## 1. 项目背景

在大模型问答场景中，如果只直接向模型提问，模型虽然能生成自然语言回答，但容易出现以下问题：

- 回答不基于指定知识库
- 容易胡编
- 不能追溯来源
- 知识更新不方便

因此，本项目使用 RAG 思路，将“本地知识库检索”与“大模型生成能力”结合起来，让模型尽量基于检索到的材料回答问题，并返回引用来源。

这个项目的目标不是做一个复杂框架，而是从零实现一个最小可用的 RAG 服务，帮助理解完整流程，并为后续扩展打基础。

---

## 2. 功能列表

当前项目支持以下能力：

### 2.1 文档处理
- 读取 `data/raw/` 下的原始文档
- 清洗并保存为结构化文档数据
- 将长文档切分为多个 chunk

### 2.2 向量化
- 使用阿里云百炼 embedding 接口对 chunk 批量向量化
- 将 embedding 结果保存到本地 JSON 文件

### 2.3 检索
- 输入 query 后，对 query 进行 embedding
- 使用 cosine similarity 做 first-stage retrieval
- 返回最相关的 top-k chunk

### 2.4 rerank 精排
- 对 embedding 初召回结果做 rerank
- 提升最终用于生成的上下文质量

### 2.5 生成回答
- 将检索结果拼接成上下文
- 调用千问模型生成最终答案
- 要求模型尽量只根据材料回答

### 2.6 API 接口
- `POST /ask`：RAG 问答
- `POST /search`：查看检索与 rerank 结果
- `POST /rebuild_index`：重建知识库索引
- `GET /ping`：健康检查
- `GET /`：项目信息

---

## 3. 技术栈

本项目主要使用：

- **Python**
- **FastAPI**
- **Uvicorn**
- **Pydantic**
- **阿里云百炼 DashScope**
- **千问大模型**
- **Embedding 接口**
- **Rerank 接口**

当前使用的核心模型：

- 生成模型：`qwen3-max-2026-01-23`
- 向量模型：`text-embedding-v4`
- rerank 模型：`qwen3-rerank`

---

## 4. 系统架构

项目当前采用“数据层 + 服务层 + 接口层”结构：

```text
qwen_rag_project/
├─ data/
│  ├─ raw/                  # 原始文档
│  ├─ processed/            # 清洗后的文档
│  ├─ chunks/               # 切分后的 chunk
│  └─ embeddings/           # 本地 embedding 索引
├─ services/
│  ├─ document_service.py   # 文档读取 / chunk 读取
│  ├─ embedding_service.py  # embedding 相关逻辑
│  ├─ retrieval_service.py  # 相似度检索
│  ├─ rerank_service.py     # rerank 精排
│  ├─ generation_service.py # 生成答案
│  └─ index_service.py      # 索引重建
├─ schemas.py               # 请求体 / 响应体模型
├─ main.py                  # FastAPI 入口
└─ README.md
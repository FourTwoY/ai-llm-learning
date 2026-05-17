# Resume Description

## 简历项目名称

基于 Qwen 的轻量级 RAG 知识库问答系统

## 简历版项目描述

- 基于 FastAPI 构建轻量级 RAG 后端服务，支持本地知识库问答与引用来源返回。
- 实现文档读取、chunk 切分、embedding、向量检索、hybrid retrieval、rerank 和回答生成链路。
- 使用 query rewrite 提升检索 query 表达质量，并通过统一配置、日志和异常处理提升可维护性。
- 使用 pytest 对配置、文档处理、检索和 query rewrite 等核心模块进行基础测试。

## 面试时 1 分钟介绍版本

这个项目是我为了系统理解 RAG 后端链路做的一个轻量级知识库问答系统。后端用 FastAPI 提供 `/rebuild_index`、`/ask` 和 `/search` 接口，模型侧接入阿里云百炼 / Qwen。离线阶段会读取本地文档、切分 chunk、生成 embedding 并保存成本地索引；在线阶段会对用户问题做 query rewrite，再进行向量检索、hybrid retrieval、rerank，最后把检索结果作为上下文交给 Qwen 生成回答。项目还补充了配置管理、日志、异常处理和 pytest 测试，方便展示完整链路和后续继续扩展。

## 面试可能被问到的问题与回答要点

### 1. 为什么要做 RAG，而不是直接问大模型？

回答要点：

- 直接问模型时，回答不一定基于指定资料。
- RAG 可以先从本地知识库检索相关内容，再让模型基于上下文回答。
- 这样能增强可解释性，也方便返回引用来源。

### 2. 离线建库流程是什么？

回答要点：

- 读取 `data/raw/` 下的 `.md` / `.txt` 文档。
- 清洗并保存 processed docs。
- 按 `chunk_size` 和 `overlap` 切分 chunk。
- 调用 embedding 模型生成向量。
- 保存到本地 `data/embeddings/all_embeddings.json`。

### 3. 在线问答流程是什么？

回答要点：

- 接收用户问题。
- 可选 query rewrite，让问题更适合检索。
- 加载本地 embedding 索引并做向量检索。
- 可选 hybrid retrieval，将向量分和关键词分加权融合。
- 可选 rerank，对候选 chunk 精排。
- 组装上下文并调用 Qwen 生成回答。
- 返回答案和引用来源。

### 4. Query rewrite 的作用是什么？

回答要点：

- 用户问题通常偏口语化，可能不适合直接检索。
- Query rewrite 会把问题改写成更清晰、更像搜索 query 的表达。
- 如果 LLM 改写失败，项目中保留了回退逻辑，避免主链路中断。

### 5. Hybrid retrieval 是怎么做的？

回答要点：

- 项目中是简化版 hybrid retrieval。
- 一路使用 embedding 做语义相似度召回。
- 一路使用关键词匹配补充显式词命中。
- 最后通过 `vector_weight` 和 `keyword_weight` 做加权融合。

### 6. Rerank 为什么放在 retrieval 后面？

回答要点：

- retrieval 负责从较大候选集中快速召回。
- rerank 负责对候选结果做更细粒度排序。
- 这样能在召回效率和最终上下文质量之间做平衡。

### 7. 项目目前的不足是什么？

回答要点：

- 当前 hybrid retrieval 还是简化实现，还没有接入标准 BM25。
- 评测集还不够系统，后续可以增加固定问题集和检索指标。
- 前端展示还比较弱，目前主要通过 Swagger UI 和 API 示例展示。

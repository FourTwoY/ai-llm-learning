# qwen_rag_project

一个基于 **FastAPI + 阿里云百炼（千问）** 的轻量级 RAG（Retrieval-Augmented Generation，检索增强生成）项目。

这个项目不是直接堆框架，而是按照“**先跑通主链路，再逐步工程化**”的思路，一天一天迭代出来的：从最基础的文档读取、切分、向量化、检索、rerank、生成回答开始，逐步补上 **参数配置化、query rewrite、hybrid retrieval、日志、异常处理、pytest 测试** 等能力。

它的定位不是“大而全的 RAG 框架”，而是一个 **结构清晰、链路完整、便于学习和展示** 的 RAG 后端项目。

---

## 1. 项目背景

在直接向大模型提问时，虽然模型可以生成自然语言回答，但通常会遇到这些问题：

- 回答不一定基于指定知识库
- 容易出现“看起来合理，但实际上不准确”的内容
- 很难追溯回答依据
- 知识更新不方便

因此，本项目采用 RAG 思路：

> 先从本地知识库中检索与问题相关的内容，再把这些内容作为上下文交给大模型生成回答。

这样做的目标是：

- 尽量让回答基于材料，而不是模型自由发挥
- 让回答过程更可解释
- 让知识更新变得更容易
- 为后续做更复杂的 RAG 系统打基础

---

## 2. 技术栈

本项目主要使用：

- **Python**
- **FastAPI**
- **Uvicorn**
- **Pydantic**
- **PyYAML**
- **Pytest**
- **阿里云百炼 DashScope / OpenAI 兼容接口**

当前使用的核心模型：

- **生成模型**：`qwen3-max-2026-01-23`
- **向量模型**：`text-embedding-v4`
- **Rerank 模型**：`qwen3-rerank`
- **Query Rewrite 模型**：`qwen3-max-2026-01-23`

---

## 3. 项目目录结构

```text
qwen_rag_project/
├─ data/
│  ├─ raw/                # 原始文档
│  ├─ processed/          # 清洗后的文档
│  ├─ chunks/             # 切分后的 chunk
│  └─ embeddings/         # 本地 embedding 索引
├─ services/
│  ├─ document_service.py         # 文档读取 / 保存 / chunk 读取
│  ├─ embedding_service.py        # embedding 生成、保存、加载
│  ├─ retrieval_service.py        # 向量检索
│  ├─ hybrid_retrieval_v1.py      # hybrid retrieval（向量 + 关键词）
│  ├─ rerank_service.py           # rerank 精排
│  ├─ generation_service.py       # 回答生成
│  ├─ query_rewrite_service.py    # query rewrite
│  ├─ logger_service.py           # 统一日志
│  ├─ exceptions.py               # 统一业务异常
│  └─ index_service.py            # 索引重建
├─ tests/                         # pytest 基础测试
├─ config.py                      # 配置读取与默认配置合并
├─ config.yaml                    # 项目参数配置
├─ schemas.py                     # 请求体 / 响应体模型
├─ main.py                        # FastAPI 入口
└─ README.md
```

---

## 4. 系统架构图

### 4.1 离线建库链路

```text
原始文档（data/raw）
    ↓
read_raw_documents()
    ↓
文档清洗 / 保存 processed docs
    ↓
build_chunks()
    ↓
chunk 文本
    ↓
build_chunk_embeddings()
    ↓
保存到 data/embeddings/all_embeddings.json
```

### 4.2 在线问答链路

```text
用户原问题
    ↓
query rewrite（可选）
    ↓
embedding retrieval
    +
keyword matching
    ↓
hybrid retrieval（可选）
    ↓
rerank（可选）
    ↓
generate answer
    ↓
FastAPI 接口返回
```

---

## 5. 当前已实现的核心能力

### 5.1 文档处理

- 读取 `data/raw/` 下的 `.txt` / `.md` 文档
- 跳过空文档
- 保存清洗后的结构化文档
- 保存切分后的 chunk 数据

### 5.2 Chunk 切分

- 支持 `chunk_size`
- 支持 `overlap`
- chunk 参数已配置化，可通过 `config.yaml` 调整

### 5.3 向量化

- 使用阿里云百炼 embedding 接口
- 对 chunk 批量向量化
- embedding 结果保存到本地 JSON
- embedding 批大小支持配置，且对接口上限做了兜底

### 5.4 检索

- 支持基础向量检索（cosine similarity）
- 支持 `top_k` 控制召回数量

### 5.5 Query Rewrite

- 在检索前先对用户原问题做一次改写
- 目标是让 query 更明确、更像搜索 query、去掉废话
- LLM 改写失败时可回退到规则清洗

### 5.6 Hybrid Retrieval

- 在向量检索基础上增加关键词匹配通道
- 通过 `vector_weight` 和 `keyword_weight` 做加权融合
- 当前是简化版 hybrid retrieval，重点在于建立“语义召回 + 关键词召回”的工程骨架

### 5.7 Rerank

- 对初召回结果做 rerank 精排
- 使用 `rerank_top_n` 控制最终送入生成模型的 chunk 数量

### 5.8 回答生成

- 将最终检索结果拼接成上下文
- 使用千问生成模型输出答案
- 尽量要求模型基于参考材料回答

### 5.9 日志与异常处理

- 补充了统一 logger
- 每个关键步骤会记录：
  - 开始时间
  - 输入摘要
  - 返回条数
  - 耗时
  - 错误信息
- 补充了统一业务异常类型：
  - 配置错误
  - 请求参数错误
  - 数据为空
  - embedding 调用失败
  - rerank 调用失败
  - generation 失败
  - 索引重建失败

### 5.10 基础测试

当前已补充 pytest 基础单元测试，覆盖了以下方向：

- 文档读取函数
- chunk 切分函数
- 检索函数
- query rewrite 函数
- 配置读取函数

这意味着这个项目已经不只是“能跑 demo”，而是开始具备最基本的可验证性。

---

## 6. 配置说明

项目中的关键参数统一由 `config.yaml` 管理。

一个典型的配置文件如下：

```yaml
models:
  generation: "qwen3-max-2026-01-23"
  embedding: "text-embedding-v4"
  rerank: "qwen3-rerank"
  rewrite: "qwen3-max-2026-01-23"

chunking:
  chunk_size: 500
  overlap: 100

embedding:
  dimension: 1024
  batch_size: 10

retrieval:
  top_k: 5
  rerank_top_n: 3
  use_hybrid: true
  vector_weight: 0.7
  keyword_weight: 0.3

rewrite:
  use_rewrite: true

logging:
  level: "INFO"

paths:
  raw_dir: "data/raw"
  processed_file: "data/processed/docs.json"
  chunks_file: "data/chunks/chunks.json"
  embeddings_file: "data/embeddings/all_embeddings.json"
```

配置化的意义是：

- 改 chunk 策略时，不改主逻辑代码
- 改检索参数时，不改主逻辑代码
- 改模型名时，不改主逻辑代码
- 后续做实验时更方便

---

## 7. 启动方式

### 7.1 环境准备

建议使用 Python 3.10+。

先安装依赖：

```bash
pip install fastapi uvicorn openai dashscope pydantic pyyaml pytest
```

### 7.2 配置 API Key

项目依赖阿里云百炼接口，需要先配置环境变量：

#### Windows PowerShell
```powershell
$env:DASHSCOPE_API_KEY = "你的千问APIKey"
```

#### Linux / macOS
```bash
export DASHSCOPE_API_KEY="你的千问APIKey"
```

### 7.3 启动服务

在 `qwen_rag_project` 根目录下执行：

```bash
uvicorn main:app --reload
```

启动成功后默认访问：

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

---

## 8. API 示例

### 8.1 健康检查

#### 请求
```http
GET /ping
```

#### 返回
```json
"pong"
```

---

### 8.2 重建索引

#### 请求
```http
POST /rebuild_index
```

#### 示例返回
```json
{
  "message": "索引重建完成",
  "doc_count": 5,
  "chunk_count": 18,
  "embedding_count": 18,
  "processed_file": "data/processed/docs.json",
  "chunks_file": "data/chunks/chunks.json",
  "embeddings_file": "data/embeddings/all_embeddings.json"
}
```

---

### 8.3 问答接口 `/ask`

#### 请求示例
```json
{
  "question": "BERT 和 GPT 的核心区别是什么？",
  "top_k": 5,
  "use_rerank": true,
  "use_rewrite": true,
  "use_hybrid": true,
  "vector_weight": 0.7,
  "keyword_weight": 0.3
}
```

#### 返回示例
```json
{
  "original_question": "BERT 和 GPT 的核心区别是什么？",
  "rewritten_query": "BERT 和 GPT 的核心区别",
  "answer": "......",
  "references": [
    {
      "source": "bert.md",
      "chunk_id": "bert_chunk_0",
      "score": 0.91
    }
  ]
}
```

---

### 8.4 检索调试接口 `/search`

这个接口更适合做“可解释链路”调试。

#### 请求示例
```json
{
  "question": "ViT 为什么能处理图像？",
  "top_k": 5,
  "use_rerank": true,
  "use_rewrite": true,
  "use_hybrid": true,
  "vector_weight": 0.7,
  "keyword_weight": 0.3
}
```

#### 返回示例
```json
{
  "original_question": "ViT 为什么能处理图像？",
  "rewritten_query": "ViT 处理图像的原因",
  "embedding_results": [
    ...
  ],
  "hybrid_results": [
    ...
  ],
  "rerank_results": [
    ...
  ]
}
```

---

## 9. 评测与调试思路

这个项目当前更适合做“工程理解型评测”，而不只是看最终答案对不对。

### 9.1 看检索链路，而不是只看最终答案

重点看：

- 召回了哪些 chunk
- 哪个 chunk 最关键
- rerank 是否把更好的块提前了
- 最终回答主要依赖了哪些证据

### 9.2 对比原问题和改写后的 query

重点看：

- rewrite 后 query 是否更简洁
- 是否更像搜索 query
- 是否更有利于召回关键 chunk

### 9.3 观察 hybrid retrieval 是否起作用

重点看：

- embedding_results 和 hybrid_results 的前几名是否发生变化
- 当 query 关键词很明确时，hybrid 是否更稳定
- 当 query 更偏语义表达时，embedding 是否仍是主力

### 9.4 做参数实验

后续可以重点对比这些参数：

- `chunk_size`
- `overlap`
- `top_k`
- `rerank_top_n`
- `vector_weight`
- `keyword_weight`

### 9.5 用 pytest 保底

每次做较大改动后，至少跑一遍：

```bash
pytest -q
```

这样可以避免：

- 配置读取被改坏
- chunk 切分逻辑回退
- query rewrite 行为异常
- 检索排序逻辑被改乱

---

## 10. 这几天的迭代记录

### Day 6：参数配置化

- 把 chunk、检索等关键参数从硬编码改为 `config.yaml` 管理
- 建立统一配置读取逻辑

### Day 10：增加 Query Rewrite

- 在检索前先对用户问题做改写
- 让 query 更适合知识库检索

### Day 11：增加 Hybrid Retrieval

- 在原有 embedding 检索基础上增加关键词通道
- 用加权方式融合

### Day 16：补日志

- 引入统一 logger
- 为关键步骤补充输入摘要、结果数量、耗时、异常日志

### Day 17：补异常处理

- 定义统一业务异常
- 提升接口报错可读性和定位效率

### Day 18：补 Pytest

- 为核心模块增加基础单元测试
- 项目从“能跑”进化到“有基础验证”

### Day 19：统一 API 返回格式

- 统一 `/ask`、`/search`、`/rebuild_index` 的响应风格
- 统一错误返回结构
- 为后续前后端联调打基础

### Day 20：完善 README

- 将项目背景、技术栈、架构、启动方式、API 示例、评测思路、优化方向系统化整理出来

---

## 11. 后续优化方向

这个项目现在已经是一个比较完整的 RAG 入门后端，但还有很多可以继续演进的方向。

### 11.1 接口层优化

- 持续完善统一 API 返回格式
- 增加更稳定的 trace_id 使用方式
- 让接口响应更规范、更利于前后端对接

### 11.2 检索优化

- 从简化版 hybrid retrieval 升级到更标准的 BM25 / 倒排索引方案
- 支持多路召回融合
- 支持更精细的字段权重

### 11.3 生成优化

- 优化 prompt 结构
- 支持更稳定的引用输出
- 进一步降低模型“自由发挥”

### 11.4 评测体系

- 建立固定问题集
- 比较不同 chunk / top_k / rerank_top_n 的表现
- 记录命中率、回答稳定性、证据质量

### 11.5 工程化能力

- 日志写入文件并按天切分
- 增加更多接口测试
- 增加 CI 检查
- 支持更方便的部署方式

---

## 12. 项目定位总结

这不是一个“大而全”的 RAG 框架项目，
而是一个 **从零理解 RAG 后端关键模块，并逐步工程化完善的学习型项目**。

它的价值在于：

- 结构清晰
- 链路完整
- 便于展示
- 便于继续扩展
- 适合作为后续更复杂 RAG 项目的基础版本


## 配置管理说明

### 1. 配置文件结构

本项目采用“配置文件 + 环境变量”方式管理配置：

- `config.yaml`：存放非敏感配置
- `.env` / `.env.dev` / `.env.prod`：存放敏感信息和环境变量
- `config.py`：统一配置入口，负责加载、合并、注入配置

### 2. 哪些内容放到哪里

#### 放在 `config.yaml` 的内容
- 模型名
- chunk 参数
- retrieval 参数
- 路径配置
- 日志级别
- 开发/生产环境差异配置

#### 放在 `.env` 的内容
- API Key
- 当前运行环境 `APP_ENV`

### 3. 环境变量示例

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
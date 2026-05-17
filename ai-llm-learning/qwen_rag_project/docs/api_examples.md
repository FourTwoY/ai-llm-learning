# API Examples

本文档记录 `qwen_rag_project` 的启动方式和常用 API 调用示例。示例中不包含真实 API Key。

## 启动服务

在 `qwen_rag_project` 目录下准备环境变量：

```env
DASHSCOPE_API_KEY=your_api_key_here
APP_ENV=dev
```

安装依赖：

```bash
pip install -r requirements.txt
```

启动 FastAPI 服务：

```bash
uvicorn main:app --reload
```

启动后访问：

- Swagger UI: `http://127.0.0.1:8000/docs`
- Root: `http://127.0.0.1:8000/`
- Ping: `http://127.0.0.1:8000/ping`

## GET /ping

用于确认服务是否正常启动。

```bash
curl http://127.0.0.1:8000/ping
```

示例返回：

```json
"pong"
```

## POST /rebuild_index

用于读取 `data/raw/` 下的文档，重新生成 processed docs、chunks 和 embedding 索引。

```bash
curl -X POST http://127.0.0.1:8000/rebuild_index
```

示例返回结构：

```json
{
  "success": true,
  "message": "索引重建完成",
  "data": {
    "doc_count": 5,
    "chunk_count": 18,
    "embedding_count": 18,
    "processed_file": "data/processed/docs.json",
    "chunks_file": "data/chunks/chunks.json",
    "embeddings_file": "data/embeddings/all_embeddings.json"
  },
  "trace_id": "example-trace-id"
}
```

## POST /ask

用于执行完整 RAG 问答链路，并返回最终答案和引用来源。

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG 的核心流程是什么？",
    "top_k": 5,
    "use_rerank": true,
    "use_rewrite": true,
    "use_hybrid": true,
    "vector_weight": 0.7,
    "keyword_weight": 0.3
  }'
```

示例返回结构：

```json
{
  "success": true,
  "message": "问答成功",
  "data": {
    "original_question": "RAG 的核心流程是什么？",
    "rewritten_query": "RAG 核心流程",
    "answer": "...",
    "references": [
      {
        "source": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md",
        "chunk_id": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_chunk_0",
        "score": 0.91
      }
    ]
  },
  "trace_id": "example-trace-id"
}
```

## POST /search

用于调试检索链路，会返回 embedding retrieval、hybrid retrieval、rerank 的中间结果。

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Transformer 为什么适合处理序列建模？",
    "top_k": 5,
    "use_rerank": true,
    "use_rewrite": true,
    "use_hybrid": true,
    "vector_weight": 0.7,
    "keyword_weight": 0.3
  }'
```

示例返回结构：

```json
{
  "success": true,
  "message": "检索成功",
  "data": {
    "original_question": "Transformer 为什么适合处理序列建模？",
    "rewritten_query": "Transformer 序列建模优势",
    "embedding_results": [],
    "hybrid_results": [],
    "rerank_results": []
  },
  "trace_id": "example-trace-id"
}
```

## Python requests 示例

```python
import requests

base_url = "http://127.0.0.1:8000"

ping_resp = requests.get(f"{base_url}/ping")
print(ping_resp.json())

ask_payload = {
    "question": "RAG 的核心流程是什么？",
    "top_k": 5,
    "use_rerank": True,
    "use_rewrite": True,
    "use_hybrid": True,
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
}

ask_resp = requests.post(f"{base_url}/ask", json=ask_payload)
print(ask_resp.json())
```

## 注意事项

- `/rebuild_index` 会调用 embedding API，需要有效的 `DASHSCOPE_API_KEY`。
- `/ask` 和 `/search` 默认会使用 query rewrite、embedding、rerank、generation 等模型能力，也需要有效的 API Key。
- 如果本地 embedding 索引不存在，`/ask` 和 `/search` 会尝试自动生成索引。

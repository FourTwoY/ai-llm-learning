# Demo Checklist

这个 checklist 用于正式展示 `qwen_rag_project` 前的手动检查和截图准备。不要提交假的截图文件，截图建议在本地跑通后手动保存。

## Demo 前准备

- 确认 Python 环境可用，建议 Python 3.10+
- 在 `qwen_rag_project` 下安装依赖：

```bash
pip install -r requirements.txt
```

- 准备 `.env`，内容参考：

```env
DASHSCOPE_API_KEY=your_api_key_here
APP_ENV=dev
```

- 确认 `config.yaml` 中的模型名、chunk 参数、retrieval 参数符合当前展示需要。

## 放入测试文档

- 将用于演示的 `.md` 或 `.txt` 文件放入：

```text
qwen_rag_project/data/raw/
```

- 建议文档数量不用太多，优先选择内容清晰、便于提问的材料。
- 文件名尽量可读，例如 `Attention Is All You Need.md`。

## 重建索引

启动服务后调用：

```bash
curl -X POST http://127.0.0.1:8000/rebuild_index
```

检查返回中：

- `success` 是否为 `true`
- `doc_count` 是否大于 0
- `chunk_count` 是否大于 0
- `embedding_count` 是否大于 0
- `data/embeddings/all_embeddings.json` 是否生成或更新

## 访问 Swagger UI

启动服务：

```bash
uvicorn main:app --reload
```

访问：

```text
http://127.0.0.1:8000/docs
```

确认 Swagger UI 中能看到：

- `GET /ping`
- `POST /rebuild_index`
- `POST /ask`
- `POST /search`

## 测试 /ask

建议在 Swagger UI 或 curl 中测试：

```json
{
  "question": "RAG 的核心流程是什么？",
  "top_k": 5,
  "use_rerank": true,
  "use_rewrite": true,
  "use_hybrid": true,
  "vector_weight": 0.7,
  "keyword_weight": 0.3
}
```

检查返回中：

- `success` 是否为 `true`
- `data.answer` 是否有回答
- `data.references` 是否包含来源文件和 chunk_id
- `data.rewritten_query` 是否比原问题更像检索 query

## 截图用于 GitHub 展示

推荐截图清单：

1. 项目目录结构截图
2. FastAPI Swagger 页面截图
3. `/rebuild_index` 成功截图
4. `/ask` 成功回答截图
5. `pytest -q` 运行通过截图

截图建议放在 GitHub README 或后续 demo 文档中，但本 checklist 不生成任何截图文件。

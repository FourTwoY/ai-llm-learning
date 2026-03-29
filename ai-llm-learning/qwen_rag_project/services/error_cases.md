# error_cases.md

## 1. API key 缺失
- 场景：未设置 `DASHSCOPE_API_KEY`
- 影响：rewrite / embedding / generation 都可能失败
- 处理：
  - 抛出 `ConfigError`
  - 接口返回：
    - `detail`: 没有检测到 DASHSCOPE_API_KEY 环境变量。
    - `error_code`: CONFIG_ERROR

---

## 2. 文档为空
- 场景：
  - `data/raw` 中没有可用文档
  - 文档切分后没有 chunk
  - embeddings 文件不存在或为空
- 处理：
  - 抛出 `DataEmptyError`
  - 接口返回更明确的 detail，例如：
    - `data/raw 中没有可用文档`
    - `文档切分后没有得到任何 chunk`
    - `embeddings 文件为空`

---

## 3. embedding 失败
- 场景：
  - DashScope embedding 接口异常
  - 网络波动
  - 请求参数错误
- 处理：
  - 抛出 `EmbeddingError`
  - 接口返回：
    - `detail`: embedding 调用失败：xxx
    - `error_code`: EMBEDDING_ERROR

---

## 4. rerank 返回异常
- 场景：
  - rerank status_code 非 200
  - rerank 响应结构异常
- 处理：
  - 抛出 `RerankError`
  - 接口返回：
    - `detail`: rerank 返回异常：...
    - `error_code`: RERANK_ERROR

---

## 5. ask 请求参数不合法
- 场景：
  - `question` 为空
  - `vector_weight + keyword_weight != 1.0`
  - 请求体字段缺失 / 类型错误
- 处理：
  - 业务校验错误：抛出 `InvalidRequestError`
  - Pydantic 校验错误：走 `RequestValidationError`
  - 接口返回：
    - `detail`: 请求参数校验失败 / 具体错误信息
    - `error_code`: INVALID_REQUEST / REQUEST_VALIDATION_ERROR

---

## 6. generation 失败
- 场景：
  - 大模型接口失败
  - 参考材料为空
- 处理：
  - 抛出 `GenerationError`
  - 接口返回：
    - `detail`: 生成回答失败：xxx
    - `error_code`: GENERATION_ERROR

---

## 7. 未知异常
- 场景：
  - 代码 bug
  - 非预期数据结构
- 处理：
  - 走 FastAPI 全局异常处理
  - 接口返回：
    - `detail`: 服务器内部异常，请查看日志定位问题。
    - `error_code`: INTERNAL_SERVER_ERROR
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=20, description="召回候选数")
    use_rerank: bool | None = Field(default=None, description="是否启用 rerank")
    use_rewrite: bool | None = Field(default=None, description="是否启用 query rewrite")
    use_hybrid: bool | None = Field(default=None, description="是否启用 hybrid retrieval")
    vector_weight: float | None = Field(default=None, ge=0.0, le=1.0, description="向量分权重")
    keyword_weight: float | None = Field(default=None, ge=0.0, le=1.0, description="关键词分权重")


class SearchRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=20, description="初召回数量")
    use_rerank: bool | None = Field(default=None, description="是否启用 rerank")
    use_rewrite: bool | None = Field(default=None, description="是否启用 query rewrite")
    use_hybrid: bool | None = Field(default=None, description="是否启用 hybrid retrieval")
    vector_weight: float | None = Field(default=None, ge=0.0, le=1.0, description="向量分权重")
    keyword_weight: float | None = Field(default=None, ge=0.0, le=1.0, description="关键词分权重")


class ReferenceItem(BaseModel):
    source: str | None = Field(default=None, description="来源文件名")
    chunk_id: str = Field(..., description="chunk 唯一标识")
    score: float = Field(..., description="相似度分数或 rerank 分数")


class SearchResultItem(BaseModel):
    chunk_id: str = Field(..., description="chunk 唯一标识")
    source: str | None = Field(default=None, description="来源文件名")
    score: float = Field(..., description="最终排序分数")
    text: str = Field(..., description="chunk 文本内容")
    vector_score: float | None = Field(default=None, description="归一化后的向量分")
    keyword_score: float | None = Field(default=None, description="关键词分")


# ===== data payload =====

class AskData(BaseModel):
    original_question: str = Field(..., description="用户原问题")
    rewritten_query: str = Field(..., description="改写后的检索 query")
    answer: str = Field(..., description="最终回答")
    references: list[ReferenceItem] = Field(..., description="引用来源列表")


class SearchData(BaseModel):
    original_question: str = Field(..., description="用户原问题")
    rewritten_query: str = Field(..., description="改写后的检索 query")
    embedding_results: list[SearchResultItem] = Field(..., description="纯 embedding 初召回结果")
    hybrid_results: list[SearchResultItem] = Field(..., description="hybrid 检索结果；如果未启用 hybrid，则为空列表")
    rerank_results: list[SearchResultItem] = Field(..., description="对 hybrid 或 embedding 结果做 rerank 后的结果")


class RebuildIndexData(BaseModel):
    doc_count: int = Field(..., description="读取到的文档数量")
    chunk_count: int = Field(..., description="切分得到的 chunk 数量")
    embedding_count: int = Field(..., description="生成的 embedding 数量")
    processed_file: str = Field(..., description="清洗后文档保存路径")
    chunks_file: str = Field(..., description="chunk 文件保存路径")
    embeddings_file: str = Field(..., description="embedding 文件保存路径")


class ErrorData(BaseModel):
    error_code: str | None = Field(default=None, description="错误码")
    path: str | None = Field(default=None, description="出错路径")
    errors: list[dict] | None = Field(default=None, description="详细校验错误列表")


# ===== unified response envelope =====

class AskResponse(BaseModel):
    success: bool = Field(default=True, description="是否成功")
    message: str = Field(..., description="响应消息")
    data: AskData = Field(..., description="业务数据")
    trace_id: str | None = Field(default=None, description="请求追踪 ID")


class SearchResponse(BaseModel):
    success: bool = Field(default=True, description="是否成功")
    message: str = Field(..., description="响应消息")
    data: SearchData = Field(..., description="业务数据")
    trace_id: str | None = Field(default=None, description="请求追踪 ID")


class RebuildIndexResponse(BaseModel):
    success: bool = Field(default=True, description="是否成功")
    message: str = Field(..., description="响应消息")
    data: RebuildIndexData = Field(..., description="业务数据")
    trace_id: str | None = Field(default=None, description="请求追踪 ID")


class ErrorResponse(BaseModel):
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    data: ErrorData | None = Field(default=None, description="错误详情")
    trace_id: str | None = Field(default=None, description="请求追踪 ID")
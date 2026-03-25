from pydantic import BaseModel, Field


# 前端提问
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=5, ge=1, le=10, description="召回候选数，范围 1~10")
    use_rerank: bool = Field(default=True, description="是否启用 rerank 二次精排")


# 一天参考来源
class ReferenceItem(BaseModel):
    source: str | None = Field(default=None, description="来源文件名")
    chunk_id: str = Field(..., description="chunk 唯一标识")
    score: float = Field(..., description="相似度分数或 rerank 分数")


# ai回答 + 参考来源
class AskResponse(BaseModel):
    answer: str = Field(..., description="最终回答")
    references: list[ReferenceItem] = Field(..., description="引用来源列表")


# 重建索引结果
class RebuildIndexResponse(BaseModel):
    message: str = Field(..., description="重建结果说明")
    doc_count: int = Field(..., description="读取到的文档数量")
    chunk_count: int = Field(..., description="切分得到的 chunk 数量")
    embedding_count: int = Field(..., description="生成的 embedding 数量")
    processed_file: str = Field(..., description="清洗后文档保存路径")
    chunks_file: str = Field(..., description="chunk 文件保存路径")
    embeddings_file: str = Field(..., description="embedding 文件保存路径")


class SearchRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=10, ge=1, le=20, description="embedding 初召回数量，范围 1~20")
    use_rerank: bool = Field(default=True, description="是否启用 rerank 二次精排")


class SearchResultItem(BaseModel):
    chunk_id: str = Field(..., description="chunk 唯一标识")
    source: str | None = Field(default=None, description="来源文件名")
    score: float = Field(..., description="相似度分数或 rerank 分数")
    text: str = Field(..., description="chunk 文本内容")


class SearchResponse(BaseModel):
    embedding_results: list[SearchResultItem] = Field(..., description="embedding 初召回结果")
    rerank_results: list[SearchResultItem] = Field(..., description="rerank 后结果；如果未启用 rerank，则为空列表")


# 报错信息
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误信息")
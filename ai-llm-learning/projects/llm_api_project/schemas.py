from pydantic import BaseModel, Field


class PaperRequest(BaseModel):
    text: str = Field(..., description="需要分析的论文摘要文本")
    style: str = Field(default="bullet", description="输出风格，例如 bullet / concise")
    max_points: int = Field(default=5, ge=1, le=10, description="最多输出几点，范围 1~10")


class PaperResponse(BaseModel):
    topic: str = Field(..., description="论文主题")
    research_problem: str = Field(..., description="论文试图解决的核心问题")
    method: str = Field(..., description="论文采用的方法")
    contributions: list[str] = Field(..., description="论文的主要贡献或创新点")
    limitations: list[str] = Field(..., description="论文的主要局限性")
    keywords: list[str] = Field(..., description="论文关键词列表")


class KeywordRequest(BaseModel):
    text: str = Field(..., description="需要提取关键词的原始文本")
    top_k: int = Field(default=5, ge=1, le=10, description="返回关键词数量，范围 1~10")


class KeywordResponse(BaseModel):
    keywords: list[str] = Field(..., description="提取出的关键词列表")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误信息说明")
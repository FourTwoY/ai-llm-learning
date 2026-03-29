class AppError(Exception):
    """
    项目统一业务异常基类
    """
    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ConfigError(AppError):
    def __init__(self, message: str = "配置错误"):
        super().__init__(message=message, code="CONFIG_ERROR", status_code=500)


class InvalidRequestError(AppError):
    def __init__(self, message: str = "请求参数不合法"):
        super().__init__(message=message, code="INVALID_REQUEST", status_code=400)


class DataEmptyError(AppError):
    def __init__(self, message: str = "数据为空"):
        super().__init__(message=message, code="DATA_EMPTY", status_code=400)


class EmbeddingError(AppError):
    def __init__(self, message: str = "embedding 调用失败"):
        super().__init__(message=message, code="EMBEDDING_ERROR", status_code=502)


class RerankError(AppError):
    def __init__(self, message: str = "rerank 调用失败"):
        super().__init__(message=message, code="RERANK_ERROR", status_code=502)


class GenerationError(AppError):
    def __init__(self, message: str = "生成回答失败"):
        super().__init__(message=message, code="GENERATION_ERROR", status_code=502)


class IndexBuildError(AppError):
    def __init__(self, message: str = "索引重建失败"):
        super().__init__(message=message, code="INDEX_BUILD_ERROR", status_code=500)
import re
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
RAG_PROJECT_DIR = BASE_DIR / "qwen_rag_project"
SEARCH_FILE_PATTERNS = ("README.md", "*.md", "*.py", "*.yaml", "*.yml")


def get_current_time() -> str:
    """
    返回当前本地时间字符串。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _extract_keywords(query: str) -> list[str]:
    normalized_query = query.lower()
    raw_keywords = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", normalized_query)
    return [keyword for keyword in raw_keywords if len(keyword.strip()) >= 2]


def _iter_searchable_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SEARCH_FILE_PATTERNS:
        files.extend(RAG_PROJECT_DIR.glob(pattern))
    return sorted({file_path for file_path in files if file_path.is_file()})


def _build_snippet(content: str, keyword: str, radius: int = 60) -> str:
    lowered_content = content.lower()
    index = lowered_content.find(keyword.lower())
    if index == -1:
        return content[: radius * 2].replace("\n", " ").strip()

    start = max(0, index - radius)
    end = min(len(content), index + len(keyword) + radius)
    return content[start:end].replace("\n", " ").strip()


def search_local_docs(query: str) -> str:
    """
    在 qwen_rag_project 里做一个极简本地关键词检索。

    说明：
    - 这里复用了现有 RAG 项目的本地文件作为检索语料。
    - 为了保持 Day 31 demo 足够轻量，这里没有直接接完整 embedding / rerank 链路，
      而是先用最小关键词匹配版本模拟“本地搜索”工具。
    """
    normalized_query = query.strip()
    if not normalized_query:
        return "本地文档检索失败：query 不能为空。"

    if not RAG_PROJECT_DIR.exists():
        return f"本地文档检索失败：未找到目录 {RAG_PROJECT_DIR}"

    keywords = _extract_keywords(normalized_query)
    if not keywords:
        return "本地文档检索失败：未能从 query 中提取有效关键词。"

    scored_results = []
    for file_path in _iter_searchable_files():
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        lowered_content = content.lower()
        matched_keywords = [keyword for keyword in keywords if keyword in lowered_content]
        if not matched_keywords:
            continue

        snippet = _build_snippet(content, matched_keywords[0])
        scored_results.append(
            {
                "path": file_path.relative_to(BASE_DIR),
                "score": len(matched_keywords),
                "matched_keywords": matched_keywords,
                "snippet": snippet,
            }
        )

    if not scored_results:
        return f"没有在本地文档中找到与“{normalized_query}”相关的内容。"

    scored_results.sort(key=lambda item: (-item["score"], str(item["path"])))
    top_results = scored_results[:3]

    lines = [f"本地文档检索结果（query: {normalized_query}）:"]
    for index, item in enumerate(top_results, start=1):
        lines.append(
            f"{index}. 文件: {item['path']} | 命中关键词: {', '.join(item['matched_keywords'])}"
        )
        lines.append(f"   摘要: {item['snippet']}")

    return "\n".join(lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "获取当前本地时间。"
                "当用户询问现在几点、当前时间、现在是什么时候时使用。"
                "不要用它回答文档检索、知识查询、资料查找类问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_docs",
            "description": (
                "搜索本地项目文档，适用于用户要求查找本地资料、README、代码注释、RAG 说明、chunk 相关内容时。"
                "当问题明显是在问时间时不要调用它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在本地文档中检索的关键词或问题。",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "search_local_docs": search_local_docs,
}

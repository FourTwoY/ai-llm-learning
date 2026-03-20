import os
import json
import dashscope


PAPER_REQUIRED_FIELDS = [
    "topic",
    "research_problem",
    "method",
    "contributions",
    "limitations",
    "keywords",
]

KEYWORD_REQUIRED_FIELDS = [
    "keywords",
]


def get_api_key():
    """
    获取 DashScope API Key
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")
    return api_key


def call_qwen_for_json(messages: list):
    """
    通用千问 JSON 调用函数
    """
    api_key = get_api_key()

    response = dashscope.Generation.call(
        api_key=api_key,
        model="qwen3-max-2026-01-23",
        messages=messages,
        result_format="message",
        response_format={"type": "json_object"},
    )

    return response


def extract_json_text(response) -> str:
    """
    提取模型返回的 JSON 字符串
    """
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(f"模型调用失败，返回结果：{response}")

    return response["output"]["choices"][0]["message"]["content"]


def parse_json(json_text: str) -> dict:
    """
    通用 JSON 解析
    """
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"返回结果不是合法 JSON：{e}")


def build_paper_messages(text: str, style: str = "bullet", max_points: int = 5):
    """
    构造论文分析 prompt
    """
    system_prompt = """
你是一个学术论文摘要分析助手。
请根据用户提供的论文摘要，返回严格可解析的 JSON。
不要输出 Markdown，不要输出解释文字，不要输出代码块标记，只输出 JSON。
""".strip()

    user_prompt = f"""
请阅读下面这段论文摘要，并输出 JSON，字段固定为：

{{
  "topic": "论文主题",
  "research_problem": "研究问题",
  "method": "使用的方法",
  "contributions": ["贡献1", "贡献2"],
  "limitations": ["局限1", "局限2"],
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}

要求：
1. 必须输出标准 JSON
2. 全部用中文
3. contributions 必须是数组
4. limitations 必须是数组
5. keywords 必须是数组
6. 返回风格参考：{style}
7. 如果摘要中没有直接写出局限性，可以基于内容做谨慎推断
8. 每个数组最多返回 {max_points} 条
9. JSON 之外不要输出任何多余内容

论文摘要如下：
{text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_keyword_messages(text: str, top_k: int = 5):
    """
    构造关键词提取 prompt
    """
    system_prompt = """
你是一个关键词提取助手。
请根据用户提供的文本，返回严格可解析的 JSON。
不要输出 Markdown，不要输出解释文字，不要输出代码块标记，只输出 JSON。
""".strip()

    user_prompt = f"""
请从下面文本中提取最重要的 {top_k} 个关键词，并输出 JSON。

固定格式如下：

{{
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}

要求：
1. 必须输出标准 JSON
2. keywords 必须是数组
3. 尽量提取最核心、最有代表性的关键词
4. 可以是中文或英文术语
5. JSON 之外不要输出任何多余内容

文本如下：
{text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def validate_paper_data(data: dict) -> dict:
    """
    校验论文分析结果
    """
    missing_fields = [field for field in PAPER_REQUIRED_FIELDS if field not in data]
    if missing_fields:
        raise ValueError(f"JSON 缺少字段：{missing_fields}")

    if not isinstance(data["contributions"], list):
        raise ValueError("字段 contributions 必须是列表。")

    if not isinstance(data["limitations"], list):
        raise ValueError("字段 limitations 必须是列表。")

    if not isinstance(data["keywords"], list):
        raise ValueError("字段 keywords 必须是列表。")

    return data


def validate_keyword_data(data: dict) -> dict:
    """
    校验关键词提取结果
    """
    missing_fields = [field for field in KEYWORD_REQUIRED_FIELDS if field not in data]
    if missing_fields:
        raise ValueError(f"JSON 缺少字段：{missing_fields}")

    if not isinstance(data["keywords"], list):
        raise ValueError("字段 keywords 必须是列表。")

    return data


def analyze_paper(text: str, style: str = "bullet", max_points: int = 5) -> dict:
    """
    论文摘要分析
    """
    if not text or not text.strip():
        raise ValueError("输入 text 不能为空。")

    messages = build_paper_messages(text, style, max_points)
    response = call_qwen_for_json(messages)
    json_text = extract_json_text(response)
    data = parse_json(json_text)
    return validate_paper_data(data)


def extract_keywords(text: str, top_k: int = 5) -> dict:
    """
    关键词提取
    """
    if not text or not text.strip():
        raise ValueError("输入 text 不能为空。")

    messages = build_keyword_messages(text, top_k)
    response = call_qwen_for_json(messages)
    json_text = extract_json_text(response)
    data = parse_json(json_text)
    return validate_keyword_data(data)
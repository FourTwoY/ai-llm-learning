import os
import re
from openai import OpenAI

REWRITE_MODEL = "qwen3-max-2026-01-23"


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def simple_rule_rewrite(question: str) -> str:
    """
    一个兜底的规则版 rewrite：
    - 去空白
    - 去掉明显口语废话
    - 保留原核心问题
    """
    q = question.strip()

    filler_patterns = [
        r"^\s*你好[呀吗，。！？\s]*",
        r"^\s*请问[一下呢吗，。！？\s]*",
        r"^\s*麻烦你[帮我]?[\s，。！？]*",
        r"^\s*能不能[帮我]?[\s，。！？]*",
        r"^\s*可以[帮我]?[\s，。！？]*",
        r"^\s*我想问一下[\s，。！？]*",
        r"^\s*帮我看一下[\s，。！？]*",
    ]

    for pattern in filler_patterns:
        q = re.sub(pattern, "", q)

    q = re.sub(r"\s+", " ", q).strip("，。！？；： ")

    return q or question.strip()


def rewrite_query(question: str) -> str:
    """
    用 LLM 做一次轻量 query rewrite：
    - 不改变用户真实意图
    - 去掉寒暄和废话
    - 改写成更适合检索的短句
    - 只输出改写后的 query
    """
    if not question or not question.strip():
        raise ValueError("question 不能为空。")

    client = get_client()

    system_prompt = """
你是一个 RAG 检索优化助手。
你的任务是把用户原问题改写成“更适合知识库检索的 query”。

要求：
1. 保留原始问题意图，不要改变意思
2. 去掉寒暄、废话、口语化表达
3. 改写后尽量更明确、更短、更像搜索 query
4. 不要回答问题
5. 不要补充原问题里没有的新事实
6. 只输出“改写后的 query 本身”，不要输出解释
""".strip()

    user_prompt = f"""
原问题：
{question}

请输出改写后的 query：
""".strip()

    try:
        completion = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        rewritten = completion.choices[0].message.content.strip()

        # 防止模型输出多余前缀
        rewritten = rewritten.replace("改写后的 query：", "").replace("改写后的query：", "").strip()

        if not rewritten:
            return simple_rule_rewrite(question)

        return rewritten

    except Exception:
        # LLM 改写失败时，退化到规则版
        return simple_rule_rewrite(question)
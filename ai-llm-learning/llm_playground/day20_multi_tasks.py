import os
import dashscope


def get_api_key():
    """
    获取 DashScope API Key
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")
    return api_key


def call_qwen(system_prompt: str, user_prompt: str) -> str:
    """
    通用千问调用函数
    """
    api_key = get_api_key()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = dashscope.Generation.call(
        api_key=api_key,
        model="qwen3-max-2026-01-23",
        messages=messages,
        result_format="message"
    )

    if response.get("status_code") != 200:
        raise ValueError(f"模型调用失败，返回结果：{response}")

    return response["output"]["choices"][0]["message"]["content"]


def classify_text(text: str) -> str:
    """
    对文本进行分类
    """
    system_prompt = "你是一个文本分类助手。请根据文本内容判断它属于哪一类。"

    user_prompt = f"""
请对下面文本进行分类，只能从以下类别中选择一个：
- 论文摘要
- 课程说明
- 产品介绍
- 新闻资讯
- 其他

要求：
1. 只输出一个类别名称
2. 不要输出解释

文本如下：
{text}
""".strip()

    return call_qwen(system_prompt, user_prompt)


def extract_keywords(text: str) -> str:
    """
    提取关键词
    """
    system_prompt = "你是一个关键词提取助手。"

    user_prompt = f"""
请从下面文本中提取 5 个最重要的关键词。

要求：
1. 用中文或英文短语都可以
2. 只输出关键词结果
3. 使用分点形式
4. 不要输出额外解释

文本如下：
{text}
""".strip()

    return call_qwen(system_prompt, user_prompt)


def rewrite_for_beginner(text: str) -> str:
    """
    改写为适合初学者阅读的版本
    """
    system_prompt = "你是一个面向初学者的改写助手。"

    user_prompt = f"""
请将下面这段文字改写成更适合初学者阅读的版本。

要求：
1. 用中文
2. 表达更简单、更清楚
3. 保留原意
4. 不要太长

原文如下：
{text}
""".strip()

    return call_qwen(system_prompt, user_prompt)


def summarize_in_bullets(text: str) -> str:
    """
    分点总结
    """
    system_prompt = "你是一个摘要助手，擅长把文本压缩成简洁要点。"

    user_prompt = f"""
请把下面这段文本总结成 3~5 个要点。

要求：
1. 用中文
2. 使用分点形式
3. 每一点尽量简洁
4. 不要输出额外解释

文本如下：
{text}
""".strip()

    return call_qwen(system_prompt, user_prompt)


def main():
    """
    测试 4 个多任务函数
    """
    try:
        sample_text = """
This course introduces the fundamentals of retrieval-augmented generation (RAG), large language models, and prompt engineering.
Students will learn how to build simple AI applications, process documents, design prompts, and evaluate generated outputs.
The course combines lectures, coding practice, and small projects, and is suitable for beginners with basic Python knowledge.
""".strip()

        print("=" * 60)
        print("原始文本")
        print("=" * 60)
        print(sample_text)
        print()

        print("=" * 60)
        print("1. classify_text(text)")
        print("=" * 60)
        print(classify_text(sample_text))
        print()

        print("=" * 60)
        print("2. extract_keywords(text)")
        print("=" * 60)
        print(extract_keywords(sample_text))
        print()

        print("=" * 60)
        print("3. rewrite_for_beginner(text)")
        print("=" * 60)
        print(rewrite_for_beginner(sample_text))
        print()

        print("=" * 60)
        print("4. summarize_in_bullets(text)")
        print("=" * 60)
        print(summarize_in_bullets(sample_text))
        print()

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
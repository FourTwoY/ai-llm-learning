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


def call_qwen(api_key, system_prompt, user_prompt):
    """
    调用千问模型
    """
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
        raise ValueError(f"调用失败，返回结果：{response}")

    return response["output"]["choices"][0]["message"]["content"]


def build_prompts(abstract_text):
    """
    构造 3 版 prompt
    """
    system_prompt = "你是一个面向大学生的学术阅读助手，回答要清晰、简洁、易懂。"

    prompt_v1 = f"""请总结下面这段论文摘要：

{abstract_text}
"""

    prompt_v2 = f"""请阅读下面这段论文摘要，并从以下三个方面进行总结：
1. 研究问题
2. 方法
3. 主要贡献

摘要如下：
{abstract_text}
"""

    prompt_v3 = f"""请阅读下面这段论文摘要，并用中文总结：
1. 按“研究问题、方法、贡献”三个部分输出
2. 使用分点形式
3. 每一点不超过 40 字
4. 表达清晰，适合大学生阅读

摘要如下：
{abstract_text}
"""

    return system_prompt, prompt_v1, prompt_v2, prompt_v3


def main():
    """
    Day 16:
    比较 3 版 prompt 的输出效果
    """
    try:
        api_key = get_api_key()

        abstract_text = """
This paper proposes a lightweight retrieval-augmented generation framework for educational question answering.
The method combines document retrieval, query rewriting, and large language model generation to improve answer accuracy.
Experiments on a university course dataset show that the proposed framework improves factual correctness and reduces hallucination compared with direct generation baselines.
""".strip()

        system_prompt, prompt_v1, prompt_v2, prompt_v3 = build_prompts(abstract_text)

        prompts = [
            ("第1版：一句话随便问", prompt_v1),
            ("第2版：要求研究问题、方法、贡献", prompt_v2),
            ("第3版：要求中文、分点、每点不超过40字", prompt_v3),
        ]

        for title, user_prompt in prompts:
            print("=" * 60)
            print(title)
            print("=" * 60)

            answer = call_qwen(api_key, system_prompt, user_prompt)
            print(answer)
            print()

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
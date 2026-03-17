import os
from openai import OpenAI


def main():
    """
    Day 15:
    1. 输入一个问题
    2. 调用 OpenAI Responses API
    3. 输出模型回答

    因调用gpt需要充值额度，故暂时未使用
    """

    # 1. 从环境变量读取 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：没有检测到 OPENAI_API_KEY 环境变量。")
        print("请先在终端中设置 API Key。")
        return

    # 2. 创建客户端
    client = OpenAI(api_key=api_key)

    # 3. 获取用户输入
    question = input("请输入你的问题：").strip()
    if not question:
        print("错误：问题不能为空。")
        return

    # 4. 额外风格要求
    style_prompt = "请面向大学生，用简洁、清楚、易懂的风格回答。"

    try:
        # 5. 调用 Responses API
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=style_prompt,
            input=question,
        )

        # 6. 输出模型文本
        print("\n模型回答：")
        print(response.output_text)

    except Exception as e:
        print("调用 API 失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
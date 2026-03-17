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


def get_user_question():
    """
    获取用户输入的问题
    """
    question = input("请输入你的问题：").strip()
    if not question:
        raise ValueError("问题不能为空。")
    return question


def call_qwen(api_key, question):
    """
    调用通义千问模型
    """
    messages = [
        {
            "role": "system",
            "content": "请面向大学生，用简洁、清楚、易懂的风格回答。"
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = dashscope.Generation.call(
        api_key=api_key,
        model="qwen3-max-2026-01-23",
        messages=messages,
        result_format="message"
    )
    return response


def extract_answer(response):
    """
    从返回结果中提取模型回答正文
    """
    return response["output"]["choices"][0]["message"]["content"]


def extract_usage(response):
    """
    提取 token 使用信息
    """
    return response.get("usage", {})


def main():
    """
    Day 15:
    跑通第一次千问 API 调用
    """
    try:
        api_key = get_api_key()
        question = get_user_question()

        response = call_qwen(api_key, question)

        status_code = response.get("status_code")
        if status_code != 200:
            print("调用失败！")
            print("完整返回结果：")
            print(response)
            return

        answer = extract_answer(response)
        usage = extract_usage(response)

        print("\n模型回答：")
        print(answer)

        print("\nToken 使用情况：")
        print(f"input_tokens : {usage.get('input_tokens')}")
        print(f"output_tokens: {usage.get('output_tokens')}")
        print(f"total_tokens : {usage.get('total_tokens')}")

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
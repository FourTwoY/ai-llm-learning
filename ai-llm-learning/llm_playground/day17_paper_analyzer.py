import os
from pathlib import Path
import dashscope


def get_api_key():
    """
    获取 DashScope API Key
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("没有检测到 DASHSCOPE_API_KEY 环境变量。")
    return api_key


# 带类型注解的函数定义   file_path: str提示该参数应传入字符串  ；  -> str 函数的返回值类型注解，提示函数最终会返回字符串
def read_text_from_file(file_path: str) -> str:
    """
    从 txt 文件中读取摘要文本
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if not path.is_file():
        raise ValueError(f"这不是一个文件：{file_path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("文件内容为空。")

    return content


def get_sample_abstract() -> str:
    """
    直接在代码中提供一个示例摘要
    """
    return """
This paper proposes a lightweight retrieval-augmented generation framework for educational question answering.
The method combines document retrieval, query rewriting, and large language model generation to improve answer accuracy.
Experiments on a university course dataset show that the proposed framework improves factual correctness and reduces hallucination compared with direct generation baselines.
""".strip()


def build_messages(abstract_text: str):
    """
    构造给千问模型的 messages
    """
    system_prompt = (
        "你是一个面向大学生的学术论文摘要分析助手。"
        "请基于用户提供的论文摘要，输出清晰、简洁、结构化的分析结果。"
        "如果某项信息摘要中没有直接写明，请基于摘要内容做谨慎推断，不要凭空编造。"
    )

    user_prompt = f"""
请阅读下面这段论文摘要，并按照固定结构输出分析结果。

输出要求：
1. 使用中文
2. 严格按照以下 5 个部分输出：
   - 研究问题
   - 方法
   - 创新点
   - 局限性
   - 适合进一步阅读的理由
3. 每一部分控制在 1~3 句话，尽量简洁
4. 使用清晰分点格式
5. 不要输出多余寒暄

论文摘要如下：
{abstract_text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def call_qwen(api_key: str, messages: list):
    """
    调用千问模型
    """
    response = dashscope.Generation.call(
        api_key=api_key,
        model="qwen3-max-2026-01-23",
        messages=messages,
        result_format="message"
    )

    return response


def extract_answer(response) -> str:
    """
    提取模型返回正文
    """
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(f"模型调用失败，返回结果：{response}")

    return response["output"]["choices"][0]["message"]["content"]


def extract_usage(response) -> dict:
    """
    提取 token 使用量
    """
    return response.get("usage", {})


def analyze_abstract(abstract_text: str) -> tuple[str, dict]:
    """
    封装分析流程：
    输入摘要，返回分析结果与 token 使用信息
    """
    api_key = get_api_key()
    messages = build_messages(abstract_text)
    response = call_qwen(api_key, messages)
    answer = extract_answer(response)
    usage = extract_usage(response)
    return answer, usage


def main():
    """
    主函数：
    支持两种输入方式
    1. 直接使用代码中的示例摘要
    2. 从 txt 文件读取摘要
    """
    try:
        print("=== 论文摘要分析器 ===")
        print("请选择输入方式：")
        print("1. 使用代码中的示例摘要")
        print("2. 从 txt 文件读取摘要")

        choice = input("请输入 1 或 2：").strip()

        if choice == "1":
            abstract_text = get_sample_abstract()
        elif choice == "2":
            file_path = input("请输入 txt 文件路径：").strip()
            abstract_text = read_text_from_file(file_path)
        else:
            print("输入无效，请输入 1 或 2。")
            return

        answer, usage = analyze_abstract(abstract_text)

        print("\n=== 分析结果 ===")
        print(answer)

        print("\n=== Token 使用情况 ===")
        print(f"input_tokens : {usage.get('input_tokens')}")
        print(f"output_tokens: {usage.get('output_tokens')}")
        print(f"total_tokens : {usage.get('total_tokens')}")

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
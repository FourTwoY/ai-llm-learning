import os
import json
from pathlib import Path
import dashscope


REQUIRED_FIELDS = [
    "topic",
    "research_problem",
    "method",
    "contributions",
    "limitations",
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
    注意：提示词里必须显式包含 JSON 关键词
    """
    system_prompt = """
你是一个学术论文摘要分析助手。
请根据用户提供的论文摘要，返回严格可解析的 JSON。
不要输出 Markdown，不要输出解释文字，不要输出代码块标记，只输出 JSON。
""".strip()

    user_prompt = f"""
请阅读下面这段论文摘要，并输出 JSON，对应字段固定为：

{{
  "topic": "论文主题",
  "research_problem": "研究问题",
  "method": "方法",
  "contributions": "主要贡献",
  "limitations": "局限性",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}

要求：
1. 必须输出标准 JSON
2. 必须包含以上 6 个字段
3. 全部用中文填写
4. 如果摘要没有直接写出局限性，可以基于摘要做谨慎推断，但不要胡编乱造
5. keywords 必须是 JSON 数组

论文摘要如下：
{abstract_text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_qwen_for_json(api_key: str, messages: list):
    """
    调用千问模型，要求返回 JSON
    """
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


def parse_and_validate_json(json_text: str) -> dict:
    """
    解析 JSON，并校验字段
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"返回结果不是合法 JSON：{e}")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        raise ValueError(f"JSON 缺少字段：{missing_fields}")

    if not isinstance(data["keywords"], list):
        raise ValueError("字段 keywords 必须是列表。")

    return data


def print_result(data: dict):
    """
    把结构化结果打印成整齐格式
    """
    print("\n=== 结构化分析结果 ===")
    print(f"topic            : {data['topic']}")
    print(f"research_problem : {data['research_problem']}")
    print(f"method           : {data['method']}")
    print(f"contributions    : {data['contributions']}")
    print(f"limitations      : {data['limitations']}")
    print(f"keywords         : {', '.join(data['keywords'])}")


def main():
    """
    支持两种输入方式：
    1. 使用代码中的示例摘要
    2. 从 txt 文件读取摘要
    """
    try:
        print("=== Day 18：结构化输出 JSON ===")
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

        api_key = get_api_key()
        messages = build_messages(abstract_text)
        response = call_qwen_for_json(api_key, messages)

        json_text = extract_json_text(response)

        print("\n=== 原始 JSON 字符串 ===")
        print(json_text)

        data = parse_and_validate_json(json_text)
        print_result(data)

    except Exception as e:
        print("程序运行失败！")
        print("错误信息：", e)


if __name__ == "__main__":
    main()
from openai import OpenAI

from .exceptions import ConfigError, InvalidRequestError, DataEmptyError, GenerationError
from .logger_service import log_step, log_result

from config import get_config




def get_client() -> OpenAI:
    cfg = get_config()
    api_key = cfg["dashscope"]["api_key"]
    if not api_key:
        raise ConfigError("没有检测到 DASHSCOPE_API_KEY 环境变量。")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def build_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""[参考材料 {i}]
chunk_id: {item["chunk_id"]}
source: {item.get("source")}
text: {item["text"]}
"""
        )
    return "\n\n".join(context_parts)


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    if not query.strip():
        raise InvalidRequestError("query 不能为空。")
    if not retrieved_chunks:
        raise DataEmptyError("retrieved_chunks 不能为空。")

    cfg = get_config()
    chat_model = cfg["models"]["generation"]

    with log_step("generate", query=query, chunk_count=len(retrieved_chunks)):
        try:
            client = get_client()
            context = build_context(retrieved_chunks)

            system_prompt = """
你是一个严谨的知识库问答助手。
请严格依据给定的参考材料回答问题。
如果参考材料不足以回答问题，请明确说明“根据当前检索到的材料，无法确定”。
不要编造参考材料中没有出现的事实。
回答尽量清晰、简洁。
""".strip()

            user_prompt = f"""
用户问题：
{query}

参考材料如下：
{context}

请完成以下任务：
1. 根据参考材料回答用户问题
2. 如果材料不足，请明确说明
3. 最后列出你使用到的参考来源 source
""".strip()

            completion = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )

            answer = completion.choices[0].message.content
            log_result("generate", result_count=1, extra={"answer_preview": answer[:100]})
            return answer

        except ConfigError:
            raise
        except Exception as e:
            raise GenerationError(f"生成回答失败：{e}") from e
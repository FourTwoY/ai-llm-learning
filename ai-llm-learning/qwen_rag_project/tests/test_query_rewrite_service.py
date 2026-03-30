from types import SimpleNamespace

from services import query_rewrite_service


class FakeCompletions:
    @staticmethod
    def create(model, messages, temperature):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="改写后的 query：BERT 预训练任务")
                )
            ]
        )


class FakeChat:
    completions = FakeCompletions()


class FakeClient:
    chat = FakeChat()


def test_rewrite_query_returns_cleaned_query(monkeypatch):
    monkeypatch.setattr(query_rewrite_service, "get_client", lambda: FakeClient())

    rewritten = query_rewrite_service.rewrite_query("请问一下，BERT 预训练的时候主要做了什么？")

    assert rewritten == "BERT 预训练任务"


class ErrorCompletions:
    @staticmethod
    def create(model, messages, temperature):
        raise RuntimeError("mock llm error")


class ErrorChat:
    completions = ErrorCompletions()


class ErrorClient:
    chat = ErrorChat()


def test_rewrite_query_falls_back_to_simple_rule_rewrite(monkeypatch):
    monkeypatch.setattr(query_rewrite_service, "get_client", lambda: ErrorClient())

    rewritten = query_rewrite_service.rewrite_query("麻烦你帮我看一下，GPT 为什么更适合生成任务？")

    assert "GPT" in rewritten
    assert "生成任务" in rewritten
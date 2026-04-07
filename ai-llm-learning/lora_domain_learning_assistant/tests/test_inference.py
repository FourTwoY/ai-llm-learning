from lora_domain_learning_assistant.src.inference.generator import fallback_answer
from lora_domain_learning_assistant.src.inference.predictor import DomainLearningPredictor


def test_fallback_answer_contains_instruction_keywords():
    answer = fallback_answer("解释 LoRA", "面向初学者")

    assert "解释 LoRA" in answer
    assert "问题背景" in answer


def test_predictor_uses_fallback_when_model_unavailable(monkeypatch):
    predictor = DomainLearningPredictor(lazy_load=True, enable_fallback=True)
    monkeypatch.setattr(predictor, "load", lambda: None)
    predictor.model_name = "fallback-rule-based"

    result = predictor.predict("解释 RAG", "")

    assert result["answer"]
    assert result["model"]

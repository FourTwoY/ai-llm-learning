# lora_domain_learning_assistant

## 项目背景

这是一个“基于 LoRA / QLoRA 的领域学习助手”最小可用项目，目标是在当前仓库里跑通从原始知识文档审计、instruction 数据构造、SFT 微调、推理、评测到 FastAPI 服务部署的完整闭环。任务场景面向 AI 论文/学习资料助手，强调回答结构清晰、风格稳定、适合复习和教学。

## 为什么原始数据不能直接拿来微调

`qwen_rag_project/data/raw` 里的原始语料主要是 Markdown 论文笔记，适合作为“知识来源材料”，但不适合直接当 SFT 样本。原因包括：

- 文本是“标题 + 条目 + 摘要/方法/结论”的文档结构，不是 instruction/input/output 对话格式。
- 不同文件模板高度相似，直接拼接容易造成重复样本和单一表达。
- 抽样时能观察到中文字符渲染异常风险，需要先清洗再统一输出格式。

详细审计见 [raw_data_audit.md](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/reports/raw_data_audit.md)。

## 数据构造策略

本项目先从 Markdown 文档中抽取标题、章节、摘要和关键词，再自动构造 FAQ 问答、概念解释、章节总结、知识点抽取、简短教学式回答和学习卡片。生成数据统一保存为 JSONL，字段固定为：

```json
{"instruction": "...", "input": "...", "output": "..."}
```

当前数据集定位是“用于跑通 LoRA/QLoRA 项目闭环的种子数据集”，后续应继续人工审查与扩充。

## 默认模型选择

默认基座模型是 `Qwen/Qwen2.5-0.5B-Instruct`，写在 [base.yaml](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/config/base.yaml) 中，而不是硬编码在脚本里。选择它的原因是参数量较小、指令跟随能力较好、社区常用、Transformers/PEFT 兼容度较高，更适合在普通本地环境做 LoRA smoke test 和工程验证。

## 技术栈

- Transformers
- PEFT
- TRL SFTTrainer
- FastAPI + Pydantic
- PyYAML
- pytest

## 目录结构

```text
lora_domain_learning_assistant/
  README.md
  requirements.txt
  .env.example
  config/base.yaml
  data/raw/
  data/processed/
  scripts/
  src/
  tests/
  reports/
```

## 数据准备

```bash
python lora_domain_learning_assistant/scripts/audit_raw_data.py
python lora_domain_learning_assistant/scripts/prepare_data.py
```

输出文件：

- [train.jsonl](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/train.jsonl)
- [val.jsonl](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/val.jsonl)
- [eval.jsonl](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/eval.jsonl)
- [sample_preview.md](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/sample_preview.md)

## 训练步骤

```bash
pip install -r lora_domain_learning_assistant/requirements.txt
python lora_domain_learning_assistant/scripts/train_lora.py --config lora_domain_learning_assistant/config/base.yaml
```

smoke test：

```bash
python lora_domain_learning_assistant/scripts/train_lora.py --smoke-test
```

如需 QLoRA，先确认环境安装了 bitsandbytes，再把 [base.yaml](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/config/base.yaml) 里的 `model.use_4bit` 改成 `true`。

## 推理步骤

```bash
python lora_domain_learning_assistant/scripts/run_inference.py --instruction "请解释 LoRA 的核心思想" --input "面向初学者"
python lora_domain_learning_assistant/scripts/run_inference.py --batch-file lora_domain_learning_assistant/data/processed/eval.jsonl --output-file lora_domain_learning_assistant/reports/inference_results.jsonl
python lora_domain_learning_assistant/scripts/run_inference.py --base-only --instruction "什么是 RAG？"
```

## API 使用方式

启动服务：

```bash
uvicorn lora_domain_learning_assistant.src.api.app:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

预测接口：

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"instruction\":\"请解释 LoRA 的核心思想\",\"input\":\"面向初学者\"}"
```

## 微调前后对比说明

```bash
python lora_domain_learning_assistant/scripts/evaluate_model.py
```

会生成 [before_after_compare.md](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/reports/before_after_compare.md) 和 [experiment_report.md](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/reports/experiment_report.md)。评测逻辑是在固定 eval 集上分别调用 base model 和 LoRA adapter 模型，输出参考答案、两版回答和简单对比字段。这不是严格 benchmark，更偏工程验收和可读性对照。

## Demo

```bash
python lora_domain_learning_assistant/scripts/build_demo_examples.py
```

## 基础测试

```bash
pytest lora_domain_learning_assistant/tests -q
```

## 局限性

- 当前自动生成的是种子训练集，质量受原始文档结构和清洗规则影响，不能替代人工标注/审核。
- 如果本地没有提前下载 base model，训练和真实推理会触发 Hugging Face 模型加载；离线环境下可能需要先手动准备本地模型路径并更新配置。
- Windows 环境下 bitsandbytes/4-bit 能力可能不可用，此时会自动回退到普通 LoRA 路径。
- 评测脚本的指标是轻量文本重叠指标，更适合作快速回归，不代表严格事实性和教学质量评测。

## 如何替换成用户自己的高质量数据

如果你已经有人工审核后的高质量 SFT 数据，只需要：

1. 直接替换 [train.jsonl](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/train.jsonl)、[val.jsonl](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/val.jsonl)、[eval.jsonl](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/data/processed/eval.jsonl)
2. 保持字段 schema 仍为 `instruction/input/output`
3. 如需换基座模型或 adapter 路径，修改 [base.yaml](D:/PycharmProjects/pythonProject/ai-llm-learning/lora_domain_learning_assistant/config/base.yaml)
4. 重新运行训练、评测和推理脚本

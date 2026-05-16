# ai-llm-learning

这是一个 AI / LLM 学习与项目实践仓库，用来记录我在 AI 应用开发、RAG、Agent、轻量级 LLM 训练等方向的学习过程和阶段性项目。

当前整理目标是面向 AI / LLM 暑期实习，把已有代码逐步整理成可展示、可复现、可讲解的项目经验。仓库中仍保留 Python 基础、API、论文阅读和小项目练习，但首页会优先突出已经具备完整链路的项目。

## Projects 总览

### 01 Qwen RAG Project

`qwen_rag_project/` 是当前最完整的项目：一个基于 FastAPI + Qwen / DashScope 的本地知识库 RAG 问答系统。

已实现能力包括：

- 文档读取与处理
- chunk 切分
- embedding 生成与本地索引保存
- vector retrieval
- query rewrite
- hybrid retrieval
- rerank
- answer generation
- 统一配置、日志、异常处理
- pytest 基础测试

### 02 MiniCode Learning

Planned。

后续作为 Coding Agent / Tool Use 学习项目入口，用来理解代码生成、工具调用、任务拆解和 Agent 执行链路。当前不虚构完成内容，等开始学习后再补充代码和文档。

### 03 MiniMind Learning

Planned。

后续作为轻量级 LLM 训练流程学习项目入口，用来理解 pretrain、SFT、inference、数据构造和训练评测等基础流程。当前不虚构完成内容，等完成阶段性实验后再整理。

## 当前重点项目

当前最适合作为 GitHub 展示和面试讲解的项目是：

```text
qwen_rag_project/
```

它已经覆盖一个轻量级 RAG 后端的主要链路：从本地文档读取、切分、向量化，到检索、改写、混合召回、精排和生成回答。项目还补充了配置管理、日志、异常处理和基础测试，比较适合作为暑期实习投递时的代表项目。

项目文档入口：

- [qwen_rag_project/README.md](qwen_rag_project/README.md)
- [qwen_rag_project/docs/architecture.md](qwen_rag_project/docs/architecture.md)
- [qwen_rag_project/docs/api_examples.md](qwen_rag_project/docs/api_examples.md)
- [qwen_rag_project/docs/demo_checklist.md](qwen_rag_project/docs/demo_checklist.md)
- [qwen_rag_project/docs/resume_description.md](qwen_rag_project/docs/resume_description.md)

## 仓库结构

```text
ai-llm-learning/
├─ api_and_data/                  # API 调用与数据处理练习
├─ basics/                        # Python 基础练习
├─ enterprise_knowledge_agent/    # 更进一步的企业知识库 Agent / RAG 项目
├─ fastapi_llm/                   # FastAPI + LLM 调用练习
├─ llm_playground/                # LLM API、prompt、论文分析练习
├─ lora_domain_learning_assistant/# LoRA / 领域学习助手方向实验
├─ notes/                         # 学习笔记与阶段总结
├─ paper_research_agent/          # 论文研究助手方向练习
├─ projects/                      # 小型综合练习项目
├─ qwen_agent_project/            # Qwen Agent / Tool Use 原型练习
├─ qwen_rag_project/              # 当前重点展示项目：Qwen RAG 后端
├─ README.md
└─ requirements.txt
```

## 学习路线

- Stage 1：整理并展示 Qwen RAG 项目
- Stage 2：学习 MiniCode，理解 Coding Agent / Tool Calling
- Stage 3：学习 MiniMind，理解 LLM pretrain / SFT / inference
- Stage 4：整合 RAG + Agent + MiniMind，形成完整项目组合

## Resume Keywords

Python, FastAPI, RAG, Qwen, DashScope, Embedding, Vector Retrieval, Hybrid Retrieval, Rerank, Query Rewrite, Pytest, LLM Application

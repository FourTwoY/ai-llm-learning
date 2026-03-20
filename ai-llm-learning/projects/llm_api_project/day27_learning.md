# Day 27 - 增加第二个接口：关键词提取

这个练习的目标是在原有论文摘要分析接口的基础上，再增加一个可用接口，让项目不再只有一个 endpoint。

## 当前接口

### 1. `POST /analyze`
输入论文摘要文本，返回结构化分析结果，包括：

- topic
- research_problem
- method
- contributions
- limitations
- keywords

### 2. `POST /keywords`
输入原始文本，返回关键词列表：

- keywords

## 使用的模型

- `qwen-plus`

## 使用的接口

- 阿里云百炼 DashScope Python SDK

## 项目结构

```text
llm_api_project/
├─ main.py
├─ schemas.py
├─ services/
│  └─ llm_service.py
├─ prompts/
│  └─ paper_prompt.txt
└─ README.md
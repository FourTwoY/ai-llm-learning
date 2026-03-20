# academic-text-analyzer

一个基于 **FastAPI + 千问 API（DashScope）** 的学术文本分析后端 demo。

这个项目主要用于对论文摘要或课程说明等文本进行分析，提供结构化输出、关键词提取等接口，适合作为大模型后端入门项目展示。

---

## 项目简介

本项目是一个简单的 LLM 后端服务示例，支持：

- 论文摘要分析
- 关键词提取
- 结构化 JSON 返回
- FastAPI 自动文档
- 基础错误处理

我在这个项目中把前面几周学到的内容整合在一起，包括：

- Python
- FastAPI
- Pydantic
- 千问 API 调用
- Prompt 设计
- JSON 结构化输出
- 接口封装与项目结构整理

---

## 技术栈

- Python
- FastAPI
- Uvicorn
- Pydantic
- DashScope（千问 API）

---

## 功能列表

### 1. `POST /analyze`
输入一段论文摘要，返回结构化分析结果，包括：

- `topic`
- `research_problem`
- `method`
- `contributions`
- `limitations`
- `keywords`

### 2. `POST /keywords`
输入一段文本，返回关键词列表：

- `keywords`

### 3. `GET /`
返回项目基础信息。

### 4. `GET /ping`
返回服务状态，用于健康检查。

---

## 接口说明

### `GET /`
返回示例：

```json
{
  "project_name": "academic-text-analyzer",
  "version": "0.6.0",
  "status": "running"
}
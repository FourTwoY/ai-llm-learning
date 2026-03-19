# Week 3 - LLM API + Prompt Practice

这是我第 3 周的大模型开发练习仓库，主要记录使用 **千问 API（DashScope）** 完成的一些基础练习，包括：

- 第一次跑通大模型 API 调用
- prompt 设计与对比
- 论文摘要分析器
- JSON 结构化输出
- 命令行版论文总结工具
- 多任务 Prompt 练习

---

## 这周做了什么

### Day 15：第一次调用大模型 API
- 使用千问 API 完成第一次单轮问答
- 理解 `messages` 的基本结构
- 学会通过环境变量读取 API Key

### Day 16：Prompt 设计对比
- 针对同一个任务写 3 版 prompt
- 比较不同 prompt 对输出结果的影响
- 理解角色设定、任务具体化、输出约束的重要性

### Day 17：论文摘要分析器
- 输入一段论文摘要
- 输出：
  - 研究问题
  - 方法
  - 创新点
  - 局限性
  - 适合进一步阅读的理由

### Day 18：结构化输出 JSON
- 让模型输出固定字段 JSON
- 在 Python 中解析并校验结果

### Day 19：命令行小工具
- 使用 `argparse` 实现命令行版论文总结工具
- 支持通过 txt 文件输入摘要
- 输出结果到终端和 `output.json`

### Day 20：多任务 Prompt 练习
实现了 4 个小函数：
- `classify_text(text)`
- `extract_keywords(text)`
- `rewrite_for_beginner(text)`
- `summarize_in_bullets(text)`

---

## 使用的 API

本周主要使用的是：

- **阿里云百炼 DashScope API**
- 模型：**qwen3-max-2026-01-23**

Python 调用方式基于：

- `dashscope.Generation.call(...)`

---

## 如何配置环境变量

在 PowerShell 中设置：

```powershell
$env:DASHSCOPE_API_KEY="你的真实APIKey"
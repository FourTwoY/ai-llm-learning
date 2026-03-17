# 文档信息整理工具

这是一个基于 Python 的小综合项目，用于扫描指定目录下的 `.txt` 和 `.md` 文件，并将文件信息整理成 CSV。

## 功能说明

程序会自动扫描目标目录中的文本文件，并输出以下信息：

- 文件名
- 文件完整路径
- 文件大小（字节）
- 字符数
- 前 100 个字符预览
- 文件编码

最终结果会保存为 `document_summary.csv`。

## 项目结构

```text
document_analyzer/
├─ main.py
├─ utils.py
├─ requirements.txt
└─ README.md
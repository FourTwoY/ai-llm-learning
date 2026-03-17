from pathlib import Path
import logging

def setup_logging():
    '''
    配置日志
    :return:
    '''

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  - %(levelname)s - %(message)s')

def find_target_files(folder: Path):
    """
    递归查找目录下所有 .txt 和 .md 文件。
    返回 Path 对象列表。
    """
    all_text = list(folder.glob("*.txt"))
    all_md = list(folder.glob("*.md"))

    return all_text + all_md

def read_text_with_multiple_encodings(file_path: Path):
    # 1.设置编码类型
    encodings = ["utf-8", "utf-8-sig", "utf-16", "gbk"]
    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            logging.info(f"成功读取文件：{file_path.name}, 编码：{encoding}")
            return content, encoding
        except UnicodeDecodeError:
            logging.warning(f"编码{encoding} 不适合文件：{file_path.name}")
            continue

    raise ValueError(f"无法识别文件编码：{file_path.name}")

def build_preview(text: str, max_length: int = 100):
    """
    获取前 max_length 个字符作为预览。
    同时将换行替换为空格，避免 CSV 里显示混乱。
    """
    cleaned_text = text.replace("\n", " ").replace("\r", " ")
    return cleaned_text[:max_length]

def analyze_single_file(file_path: Path):
    """
    分析单个文件，返回字典结果。
    包含：
    - 文件名
    - 路径
    - 文件大小
    - 字符数
    - 前100字符预览
    - 编码
    """
    content, encoding = read_text_with_multiple_encodings(file_path)

    return {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_size_bytes": file_path.stat().st_size,
        "char_count": len(content),
        "preview_100_chars": build_preview(content, 100),
        "encoding": encoding
    }


def analyze_folder(folder: Path):
    """
    扫描整个目录，分析其中所有 txt/md 文件。
    返回结果列表。
    """
    all_files = find_target_files(folder)

    if not all_files:
        logging.warning("没有找到任何 .txt 或 .md 文件。")
        return []

    results = []

    for file_path in all_files:
        try:
            logging.info(f"开始处理文件：{file_path}")
            file_info = analyze_single_file(file_path)
            results.append(file_info)
        except Exception as e:
            logging.error(f"处理文件失败：{file_path}，错误信息：{e}")

    return results
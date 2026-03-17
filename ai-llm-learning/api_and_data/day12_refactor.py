import os
from pathlib import Path
import csv
import logging

def setup_logging():
    '''
    配置日志格式和级别
    INFO：显示正常运行信息
    ERROR：显示错误信息
    :return:
    '''
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s -  %(levelname)s - %(message)s')

def read_text_with_multiple_encodings(file_path):
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

def count_file_info(file_path):

    '''
    统计单个文本文件的信息：
    ①文件名  ②文件大小（字节）
    ③行数   ④字符数
    :param file_path:
    :return:
    '''

    content, encoding = read_text_with_multiple_encodings(file_path)

    file_info = {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "line_count": len(content.splitlines()),
        "char_count": len(content),
        "encoding": encoding
    }

    return file_info

def find_target_files(folder):
    '''
    查找目录下所有 .txt  和  .md文件
    返回Path对象列表
    :param folder:
    :return:
    '''

    txt_files = list(folder.glob("*.txt"))
    md_files = list(folder.glob("*.md"))
    all_files = txt_files + md_files

    return all_files

def save_to_csv(results, output_file):
    '''
    将统计结果保存到csv文件
    :param results:
    :param output_file:
    :return:
    '''

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "file_size_bytes",
                "line_count",
                "char_count",
                "encoding"
            ]
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    """
    主函数：
    1. 配置日志
    2. 扫描目标文件
    3. 逐个统计文件信息
    4. 保存结果到 CSV
    """
    setup_logging()

    folder = Path("..")  # 返回上级目录
    output_file = folder / "file_stats.csv"

    logging.info(f"开始扫描目录：{folder.resolve()}")

    try:
        all_files = find_target_files(folder)

        if not all_files:
            logging.warning("当前目录下没有找到 .txt 或 .md 文件。")
            return

        results = []

        for file_path in all_files:
            try:
                logging.info(f"开始处理文件：{file_path.name}")
                file_info = count_file_info(file_path)
                results.append(file_info)
            except Exception as e:
                logging.error(f"处理文件失败：{file_path.name}，错误信息：{e}")

        if not results:
            logging.warning("没有成功处理任何文件，程序结束。")
            return

        save_to_csv(results, output_file)

        logging.info(f"处理完成，共统计 {len(results)} 个文件。")
        logging.info(f"结果已保存到：{output_file.resolve()}")

    except Exception as e:
        logging.critical(f"程序运行出现严重错误：{e}")


if __name__ == "__main__":
    main()








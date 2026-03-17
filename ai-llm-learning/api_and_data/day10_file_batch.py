from pathlib import Path
import csv


def count_file_info(file_path):

    """
    统计单个文本文件的信息：
    ①文件名  ②文件大小（字节）
    ③行数   ④字符数
    :param file_path:
    :return:
    """

    encodings = ["utf-8", "utf-8-sig", "utf-16", "gbk"]

    content = None
    used_encoding = None

    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"无法识别文件编码：{file_path.name}")

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "line_count": len(content.splitlines()),
        "char_count": len(content),
        "encoding": used_encoding
    }

def main():
    '''
    扫描当前目录下的  .txt  和  .md文件
    统计信息后保存到 file_stats.csv
    :return:
    '''

    # 指定要扫描的目录
    folder = Path("..")
    print("开始扫描目录：", folder.resolve())   # 将相对路径转化为绝对路径，返回Path类对象

    # 准备存统计结果的列表
    result = []

    # 查找 txt  md文件
    txt_files = folder.glob("*.txt")
    md_files = folder.glob("*.md")

#     合并两类文件
    all_files = list(txt_files) + list(md_files)

    if not all_files:
        print("当前目录下没有找到 .txt 或  .md文件")
        return

    for file_path in all_files:
        print(f"正在处理文件： {file_path.name}")
        try:
            file_info = count_file_info(file_path)
            result.append(file_info)
        except Exception as e:
            print(f"处理文件失败：{file_path.name}")
            print("错误信息：", e)

#   定义输出文件路径
    out_file = folder / "file_stats.csv"

    # newline=""    禁止自动换行转换
    with open(out_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "file_size_bytes", "line_count", "char_count", "encoding"]
        )
        writer.writeheader()
        writer.writerows(result)
        print("\n处理完成!")
        print(f"共统计{len(result)}个文件夹。")
        print("结果已保存到：", out_file.resolve())


if __name__ == "__main__":
    main()
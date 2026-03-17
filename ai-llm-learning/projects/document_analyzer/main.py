# main.py

from pathlib import Path
import logging
import pandas as pd

from utils import setup_logging, analyze_folder


def main():
    """
    主程序入口：
    1. 配置日志
    2. 输入要扫描的目录
    3. 调用分析函数
    4. 用 pandas 保存为 CSV
    """
    setup_logging()

    print("=== 文档信息整理工具 ===")
    print("请输入你要扫描的目录路径。")
    print("如果直接回车，则默认扫描当前项目目录。")

    user_input = input("请输入目录路径：").strip()

    if user_input:
        folder = Path(user_input)
    else:
        folder = Path("")

    if not folder.exists():
        logging.error(f"目录不存在：{folder}")
        print("错误：目录不存在，请检查路径。")
        return

    if not folder.is_dir():
        logging.error(f"输入的不是目录：{folder}")
        print("错误：你输入的路径不是文件夹。")
        return

    logging.info(f"开始扫描目录：{folder.resolve()}")

    results = analyze_folder(folder)

    if not results:
        print("没有找到可处理的 txt/md 文件，程序结束。")
        return

    df = pd.DataFrame(results)

    output_file = Path("document_summary.csv")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    logging.info(f"处理完成，共整理 {len(df)} 个文件。")
    logging.info(f"结果已保存到：{output_file.resolve()}")

    print("\n处理完成！")
    print(f"共整理 {len(df)} 个文件。")
    print(f"结果文件：{output_file.resolve()}")


if __name__ == "__main__":
    main()
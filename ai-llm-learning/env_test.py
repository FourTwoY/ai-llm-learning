import requests
import pandas as pd


def main():
    print("requests version:", requests.__version__)
    print("pandas version:", pd.__version__)

    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95, 88, 92]
    }

    df = pd.DataFrame(data)

    print("\nDataFrame 内容如下：")
    print(df)

    print("\n环境测试成功：requests 和 pandas 都可以正常导入与使用。")


if __name__ == "__main__":
    main()
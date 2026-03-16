import requests
import pandas as pd

def main():
    """
    day09 练习
    1. 请求公开测试 API
    2. 获取 JSON 数据
    3. 提取关心的字段
    4. 打印成更好读的格式
    """

    # 定义接口地址
    url = "https://jsonplaceholder.typicode.com/users"

    print("正在请求接口...")
    print(f"请求地址：{url}")

    response = requests.get(url)

    if response.status_code == 200:
        print("\n请求成功！")
        print("状态码：", response.status_code)
    else:
        print("\n请求失败！")
        print("状态码：", response.status_code)

    users = response.json()

    print("\n返回数据类型：", type(users))
    print("用户数量：", len(users))

    df = pd.DataFrame(data=users)

    # print(df)

    print("数据表格的前3行：")
    print(df[["name", "email", "phone"]].head(3))



if __name__ == "__main__":
    main()









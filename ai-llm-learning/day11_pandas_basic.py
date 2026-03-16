import pandas as pd

def main():
    '''
    ①读取score.csv    ②计算每个学生平均分
    ③找出最高分学生     ④输出新的csv文件
    :return:
    '''

    # 读csv文件
    file_path = "score.csv"
    df = pd.read_csv(file_path)

    print("原始成绩表：")
    print(df)

    # 选列
    print("\n只查看 name 和 math 两列：")
    print(df[["name", "math"]])

#     计算每个学生的平均分
    df["average"] = ((df["math"] + df["english"] + df["python"]) / 3).round(2)

    print("\n加入 average 列后的成绩表")
    print(df)

    # 过滤：找出 average >= 90 的学生
    excellent_students = df[df["average"] >= 90]

    print("\n平均分大于等于 90 的学生：")
    print(excellent_students)



#     找出平均分最高的学生
    top_student = df.loc[df["average"].idxmax()]

    print("\n平均分最高的学生：")
    print(f"姓名：{top_student['name']}")
    print(f"数学：{top_student['math']}")
    print(f"英语：{top_student['english']}")
    print(f"Python：{top_student['python']}")
    print(f"平均分：{top_student['average']:.2f}")

#     输出新的csv文件
    output_file = "scores_result.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n处理完成，结果已保存到{output_file}")


if __name__ == '__main__':
    main()



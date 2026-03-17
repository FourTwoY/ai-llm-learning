def add_student(students):
    name = input("请输入学生姓名：")
    score = float(input("请输入学生成绩："))

    student = {"name": name, "score": score}
    students.append(student)

    print("添加成功")

def find_student(students):
    name = input("请输入要查询的学生姓名：")

    for student in students:
        if student["name"] == name:
            print(f"姓名：{student['name']}, 成绩：{student['score']}")
            return

    print("未找到该学生")

def show_all_students(students):
    if len(students) == 0:
        print("当前没有学生信息")
        return

    print("所有学生信息如下:")
    for student in students:
        print(f"姓名：{student['name']}, 成绩：{student['score']}")


def show_average_score(students):
    if len(students) == 0:
        print("没有学生数据，无法计算平均分")
        return

    total = 0
    for student in students:
        total += student["score"]

    avg = total / len(students)
    print(f"所有学生的平均分为{avg:.2f}")
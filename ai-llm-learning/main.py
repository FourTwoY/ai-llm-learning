from utils import add_student, find_student, show_all_students, show_average_score

def main():
    students = []

    while True:
        print("\n=====学生成绩管理系统=====")
        print("1. 添加学生")
        print("2. 查询学生")
        print("3. 输出所有学生平均分")
        print("4. 显示所有学生")
        print("5. 退出系统")

        choice = input("请输入你的选择：")

        if choice == '1':
            add_student(students)
        elif choice == '2':
            find_student(students)
        elif choice == '3':
            show_average_score(students)
        elif choice == '4':
            show_all_students(students)
        elif choice == '5':
            print("退出系统")
            break
        else:
            print("输入无效，请重新输入")


if __name__ == '__main__':
    main()
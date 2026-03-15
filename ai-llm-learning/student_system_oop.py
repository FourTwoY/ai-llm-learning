class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def show_info(self):
        print(f"姓名: {self.name}, 成绩: {self.score}")

class StudentManager:
    def __init__(self):
        self.students = []

    def add_student(self):
        name = input("请输入学生姓名：")

        try:
            score = float(input("请输入学生成绩："))
        except ValueError:
            print("成绩输入无效")
            return

        student = Student(name, score)
        self.students.append(student)
        print("添加成功")


    def show_all_students(self):
        if len(self.students) == 0:
            print("当前没有学生信息")
            return

        print("所有学生信息如下：")
        for student in self.students:
            student.show_info()

    def find_student_by_name(self):
        if len(self.students) == 0:
            print("当前没有学生信息")
            return

        name = input("请输入要查询的学生姓名：")

        for student in self.students:
            if student.name == name:
                print("找到学生信息：")
                student.show_info()
                return

        print("未找到该学生")


    def menu(self):
        while True:
            print("\n===== 学生成绩管理系统 =====")
            print("1. 添加学生")
            print("2. 打印所有学生信息")
            print("3. 根据姓名查询学生")
            print("4. 退出系统")

            choice = input("请输入你的选择: ")

            if choice == "1":
                self.add_student()
            elif choice == "2":
                self.show_all_students()
            elif choice == "3":
                self.find_student_by_name()
            elif choice == "4":
                print("退出系统")
                break
            else:
                print("输入无效，请重新输入")

if __name__ == "__main__":
    student_manager = StudentManager()
    student_manager.menu()
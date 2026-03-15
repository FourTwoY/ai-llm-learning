# prac1
str = input("Enter a string: ")
print("所输入的字符串长度为：%d" %len(str))
print("所输入的字符串长度为：%d" %str.__len__())

#prac2
list1 = []
for i in range(0,5):
    list1.append(int(input("Enter a number: ")))

max1 = max(list1)
avg1 = sum(list1)/len(list1)
print("最高成绩：", max1)
print(f"平均成绩：{avg1: .2f}")

#prac 3
dict1 = {"jack" : {"score":98}, "rose" : {"score":65}, "alice" : {"score":87}}
def func1(score):
    match score:
        case x if x >= 90:
            return "A(优秀)"
        case x if x >= 80:
            return "B(良好)"
        case _:
            return "C(一般)"

for key,value in dict1.items():
    grade = value["score"]
    level = func1(grade)
    print(f"姓名：{key}| 分数：{grade} | 等级：{level}")
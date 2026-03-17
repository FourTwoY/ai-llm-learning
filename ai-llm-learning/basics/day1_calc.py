# 练习1：输入两个数，输出和、差、积、商
import math

print("=====四则运算=====")
a = float(input("请输入第一个数："))
b = float(input("请输入第二个数："))

sum_ab = a + b
diff_ab = a - b
mul_ab = a * b
if b != 0:
    div_ab = a / b
else:
    print("除数不能为零")

print("两数之和为：%.2f" % sum_ab)
print("两数之差为：%.2f" % diff_ab)
print("两数之积为：%.2f" % mul_ab)
print("两数之商为：%.2f" % div_ab)

# 练习2：输入半径，输出圆的面积
print("===根据半径求圆的面积===")
c = float(input("请输入圆的半径："))
area = math.pi * c ** 2
print(f"半径为{c}的圆面积为：{area:.2f}")

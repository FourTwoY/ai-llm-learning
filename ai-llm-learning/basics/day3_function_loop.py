import math


def is_prime(n):            # 判断素数
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

def factorial(n):         #  计算阶乘
    if n <= 1:
        return 1
    return factorial(n-1) * n

def calc_avg(score):
    return sum(score)/len(score)

def find_max(nums):
    i = -math.inf
    for num in nums:
        if num > i:
            i = num
    return i




a = int(input("Enter a number: "))
print(f"输入的数{a}是质数吗：{is_prime(a)}")
print(f"5的阶乘为：{factorial(5)}")

list1 = [1, 2, 3, 4, 5]
print(f"list1中的最大值为：{find_max(list1)}")

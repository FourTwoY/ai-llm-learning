L = ['Hello', 'World', 'IBM', 'Apple']

print([s.lower() for s in L])

def func(n):
    return n * n
f = func   # 给函数起别名
print(f(4))
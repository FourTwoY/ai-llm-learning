import os
from functools import reduce

try:
    line_count = 0
    word_count = 0
    char_count = 0
    char_noblank_count = 0

    with open('../notes/notes.txt', 'r') as f:  # 相对路径
        for line in f:
            print(line.strip())
            line_count += 1
            word_count += len(line.split())

            str_list = line.split()
            no_blank = len(reduce(lambda x, y: x + y, str_list))
            char_noblank_count += no_blank
            # print(f"不带空格字符数:{char_noblank_count}")
            char_count += len(line)

        print(f"行数:{line_count}")
        print(f"单词数:{word_count}")
        print(f"字符数:{char_count}")
        print(f"不带空格字符数:{char_noblank_count}")

except FileNotFoundError:
    print("错误：notes.txt文件不存在")
# print(os.name)  #nt  Windows

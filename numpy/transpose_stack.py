"""

"""
import numpy as np


# 使用.T实现转置
a = np.zeros((2,3))
print("===initial array===")
print(a)

print("===transfer T===")
print(a.T)

# 合并
# 对于二维数组而言， r_ 和 c_ 分别表示上下合并和左右合并
b = np.array([[1,1,1], [1,1,1]])
c = np.array([[2,2,2], [2,2,2]])
print("=== array1 ===")
print(b)
print("=== array2 ===")
print(c)

# 上下合并
# 使用r_
r_stack = np.r_[b, c]
print("=== vertical stack ====")
print(r_stack)
# 或者使用vstack
print(np.vstack((b, c)))

# 左右合并
# 使用c_
c_stack = np.c_[b, c]
print("=== horizontal stack ====")
print(c_stack)
# 或者使用hstack
print(np.hstack((b, c)))


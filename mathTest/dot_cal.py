"""
矩阵乘法可以使用@符号，python3.5后新增的运算符
"""
import numpy as np

# case1: 矩阵@矩阵
# 它与numpy.dot（）的作用是一样的，矩阵乘法（就是线性代数里学的）！
a = np.array([[1, 2], [1, 2]])
b = np.array([[5, 6], [5, 6]])
print("===== a@b =====")
print(a@b)
print("===== numpy.dot =====")
print(np.dot(a, b))
print(a*b)

# case2: 矩阵@向量
c = np.array([1, 2])
d = np.array([[5, 6], [5, 6], [5, 6]])
# 直接使用c@d,会报错，因为c是一个一维向量(2,)，而d是一个二维矩阵(3,2)
# print("===== c@d =====")
# print(c@d)
# 如果第二个位置放的是向量（向量可以看作是1行n列的矩阵）的话，那么向量会转置！
print("===== d@c =====")
print(d@c)


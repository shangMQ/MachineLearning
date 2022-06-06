import numpy as np
"""
np.finfo()返回某种数据类型的相关参数

eps是一个很小的非负数
除法的分母不能为0的,不然会直接跳出显示错误。
使用eps将可能出现的零用eps来替换，这样不会报错。
"""
print(np.float)
print(np.finfo(np.float))


x = np.array([1, 2, 3], dtype=float)
eps = np.finfo(x.dtype).eps
print(eps)

height = np.array([0, 2, 3], dtype=float)
# 一旦出现0， 可以用eps进行替换
height = np.maximum(height, eps)
print(height)

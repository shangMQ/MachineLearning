"""
特殊矩阵的生成：单位阵、对角阵等等
"""
import numpy as np

# 单位阵
std_mat = np.eye(3)
print("=== standard mat ===")
print(std_mat)
# 偏移主对角线1个单位的伪单位矩阵
offset_std_mat = np.eye(3, k=1)
print("=== offset mat ===")
print(offset_std_mat)

# 全填充生成矩阵
full = np.full((2,3), 10)
print("=== full mat ===")
print(full)

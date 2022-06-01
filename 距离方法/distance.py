from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin
import numpy as np

"""
利用sklearn快速计算距离
原先计算套用欧几里得计算方法，速度很慢，现在可以使用sklearn提供的库直接计算
"""

a1 = np.array([[2, 2, 1],
               [2, 3, 1],
               [3, 2, 4]])

a2 = np.array([[3, 2, 1], [1, 1, 1]])

# 计算距离
distance = pairwise_distances(a1, a2, metric="euclidean")
print(distance)
print(np.argmin(distance, axis=1))

# 计算距离最小值的索引
indexes = pairwise_distances_argmin(a1, a2, metric="euclidean")
print(indexes)
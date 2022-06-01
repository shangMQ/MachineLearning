from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin
import numpy as np
from sklearn.utils.extmath import row_norms, safe_sparse_dot

"""
利用sklearn快速计算距离
原先计算套用欧几里得计算方法，速度很慢，现在可以使用sklearn提供的库直接计算
"""


def euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None):
    """
    手动欧几里得距离计算，参考sklearn的设计
    缺点：需要计算范数，然后套用公式，时间较慢
    """
    if X_norm_squared is not None:
        XX = X_norm_squared.reshape(-1, 1)
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if Y is X:
        YY = XX.T
    else:
        if Y_norm_squared is not None:
            YY = Y_norm_squared.reshape(1, -1)
        else:
            YY = row_norms(Y, squared=True)[:, np.newaxis]

    distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
    distances += XX
    distances += YY

    return distances


a1 = np.array([[2, 2, 1],
               [2, 3, 1],
               [3, 2, 4]])

a2 = np.array([[3, 2, 1], [1, 1, 1]])

# 使用sklearn计算
print("====sklearn.pairwise_distance=====")
# 计算距离
distance = pairwise_distances(a1, a2, metric="euclidean")
print(distance)
print(np.argmin(distance, axis=1))

# 计算距离最小值的索引
indexes = pairwise_distances_argmin(a1, a2, metric="euclidean")
print(indexes)

# 手动计算
print("====manual calculate distance=====")
print(np.sqrt(euclidean_distances(a1, a2[0].reshape(1,-1))))

"""
sklearn下距离的度量

"""
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


X = np.array([[1, 2], [1, 3], [1, 4]])
Y = np.array([1, 1]).reshape(1,-1)

# 计算一个样本集内部样本之间的距离
# sklearn实现
D1 = pairwise_distances(X, X, metric='euclidean', squared=False)
print(D1)

# kmeans质心距离计算
D2 = pairwise_distances(X, Y, metric='euclidean', squared=False)
print(D2)
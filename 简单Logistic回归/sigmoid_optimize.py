"""
sigmoid对于正负数的优化
"""
import numpy as np
from scipy.special import expit

def sigmoid_naive(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_optimize(X):
    """
    手动对原公式进行修改
    """
    if X < 0:
        return np.exp(X)/(1+np.exp(X))
    else:
        return 1 / (1 + np.exp(-X))


# 当X为很大的负数时，np.exp(-X)k会产生overflow问题。示例如下：
X = 1.11
naive = sigmoid_naive(X)

# 方式一：对原公式手动进行修改
handy = sigmoid_optimize(X)

# 方式二：使用scipy.special.expit()函数
usage = expit(X)
print(f"naive = {naive}")
print(f"handy = {handy}")
print(f"scipy = {usage}")
print(f"naive and scipy, tol = {abs(naive - usage)}")
print(f"handy and scipy, tol = {abs(handy - usage)}")

# X = -10000000000
# print(sigmoid_optimize(X))
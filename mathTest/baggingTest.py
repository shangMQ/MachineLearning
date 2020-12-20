# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
n个弱分类器可以组合成一个强分类器
"""

import operator
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def c(n, k):
    """
    组合数计算方法
    :param n:
    :param k:
    :return:
    """
    return reduce(operator.mul, range(n-k+1, n+1)) / reduce(operator.mul, range(1, k+1))


def bagging(n, p):
    s = 0.0
    for i in range(n // 2 + 1, n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s


def main():
    T = np.arange(9, 100, 10)
    scoreList = []

    for i, t in enumerate(T):
        score = bagging(t, 0.6)
        scoreList.append(score)
        print("第%d次采样第正确率：%f" % (t, score))

    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.plot(T, scoreList, 'r-o')
    plt.title("Bagging")
    plt.xlabel("X")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
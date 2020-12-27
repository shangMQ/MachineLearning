# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
集成学习的简单介绍
    多个弱分类器集合成一个强分类器
"""
import operator
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb


def c(n, k):
    """
    自己生成组合数
    :param n:
    :param k:
    :return:
    """
    return reduce(operator.mul, list(range(n-k+1, n+1))) / reduce(operator.mul, list(range(1, k+1)))


def bagging(n, p):
    """
    直接使用scipy的special包计算组合数
    :param n:
    :param p:
    :return:
    """
    s = 0
    for i in range(n // 2 + 1, n + 1):
        s += comb(n, i) * p ** i * (1 - p) ** (n - i)
    return s


def main():
    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False

    n = 100
    x = np.arange(1, n, 2)
    y = np.empty_like(x, dtype=np.float)
    for i, t in enumerate(x):
        y[i] = bagging(t, 0.6)
        if t % 10 == 9:
            print(t, '个分类器的正确率：', y[i])

    plt.figure(facecolor='w')
    plt.plot(x, y, 'ro-', lw=2, mec='k')
    plt.xlim(0, n)
    plt.ylim(0.6, 1)
    plt.xlabel('The number of classifier', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('The accuracy of ensumble classifier', fontsize=20)
    plt.grid(b=True, ls=':', color='#606060')
    plt.show()


if __name__ == "__main__":
    main()
# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
统计数字的概率：
    给定某正整数N，统计从1到N的所有数的阶乘中，首位数字出现1的概率，出现2的概率...出现9的概率）

本福特定律：第一数字定律
    在实际生活中得出的一组数据，以1为首的数字出现的概率约为总数的三成，即1/9的三倍

应用：
    · 经济数据反欺诈
    · 阶乘/素数数列/斐波那契数列首位
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def first_digital(x):
    """
    获取数字x的首位数
    :param x:
    :return:
    """
    x_str = str(x)
    return int(x_str[0])


def main():
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    N = int(input("请输入想要计算的阶乘数据："))
    n = 1
    frequency = [0] * 9
    for i in range(1, N):
        n *= i
        m = first_digital(n) - 1
        frequency[m] += 1

    for i, f in enumerate(frequency):
        print("%d : %d次" % (i+1, f))

    x = np.arange(1, 10)

    plt.plot(x, frequency, 'r-', x, frequency, 'go', lw=2, markersize=8)
    plt.xlabel("Number")
    plt.ylabel("Presence Times")
    plt.title("The first number of Factorial(n), 1 <= n <= 999 ")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()


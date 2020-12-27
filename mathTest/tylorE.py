# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS

"""
利用泰勒公式，求ex
"""
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn


def calc_e_small(x):
    """
    使用泰勒展开式计算的ex在x0=0时刻的展开公式
    e**x = 1 + x + x**2 / 2! + x**3 / 3! + ... + x**n / n!
    :param x:
    :return:
    """
    n = 10  # 假设目前计算前n项
    f = np.arange(1, n+1).cumprod()  # 累乘得到每个分母
    b = np.array([x ** i for i in range(1, n+1)])  # 累乘得到分子x
    return np.sum(b / f) + 1


def calc_e(x):
    reverse = False
    if x < 0:   # 处理负数
        x = -x
        reverse = True
    ln2 = 0.69314718055994530941723212145818
    # 将一个数拆成两部分a+b
    # 其中c是x/ln2,求出系数a，注意a是整数
    c = x / ln2
    a = int(c+0.5)
    # b是x-aln2，是那个余数
    b = x - a*ln2
    y = (2 ** a) * calc_e_small(b)
    if reverse:
        return 1/y
    return y


def main():
    # 1. 设置字体
    mpl.rcParams['font.sans-serif'] = [u'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False

    # 2. 生成数据
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 4, 20)
    t = np.concatenate((t1, t2))
    print(t)  # 横轴数据
    y = np.empty_like(t)

    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, ' = ', y[i], '(近似值)\t', math.exp(x), '(真实值)')

    plt.figure(facecolor='w')
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2, markeredgecolor='k')
    plt.title('Taylor展式的应用 - 指数函数', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True, ls=':')
    plt.show()


if __name__ == "__main__":
    main()
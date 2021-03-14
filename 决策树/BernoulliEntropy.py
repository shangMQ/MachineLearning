# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
伯努利信息熵
分布为伯努利分布时熵与概率的关系
验证：
0<= H(p) <= log(n)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def calculateH(p):
    """
    信息熵计算
    :param p: 概率
    :return:
    """
    return -(p * np.log2(p)) - (1-p) * np.log2(1-p)


def main():
    # 设置图像的标题字体
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 生成[0,1]之间的概率值
    p = np.linspace(0, 1, 100)
    # print(p)

    # 2. 计算伯努利信息熵
    Hp = calculateH(p)

    # 3. 绘图
    plt.plot(p, Hp, linewidth=2)
    plt.scatter([0.5], [1.0], color='r', edgecolor='k', s=50)
    plt.axvline(0.5, ls="--", color='gray', linewidth=1)
    plt.axhline(1, ls="--", color='gray', linewidth=1)
    plt.xlabel("p")
    plt.ylabel("H(p)")
    plt.title("Entropy changes with p under Bernoulli Distribution")
    plt.show()


if __name__ == "__main__":
    main()


# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
np.random的使用
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def main():
    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 生成数据
    data = 2 * np.random.rand(1000, 2) - 1
    x = data[:, 0]
    y = data[:, 1]

    # 2. 计算单位圆的索引
    idx = x ** 2 + y ** 2 < 1

    # 3. 戳一个洞
    hole = x ** 2 + y ** 2 < 0.25

    # 4. 进行逻辑操作
    idx = np.logical_and(idx, ~hole)

    plt.plot(x[idx], y[idx], 'ro', markersize=1)
    # plt.title("Bagging")
    # plt.xlabel("X")
    # plt.ylabel("Accuracy")
    # plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
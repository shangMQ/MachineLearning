# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
利用梯度下降算法计算一个数的平方根
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = u'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False
    # 1. 设置学习率
    learning_rate = 0.01

    # 2. x的值
    x = np.arange(1, 100)
    y_predict = []
    y_true = np.sqrt(x)

    # 3. 利用梯度下降法计算平方根
    for num in x:
        cur = 0

        for i in range(1000):
            cur -= learning_rate * (cur**2 - num)

        y_predict.append(cur)
        # print("%d的平方根计算，近似值 = %f, 真实值 = %f" % (num, cur, math.sqrt(num)))

    # 4. 绘制图像
    plt.plot(x, y_predict, 'r--', alpha=0.6, label="GDA")
    plt.plot(x, y_true, 'g-', alpha=0.6, label="Math")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$y = \sqrt{x}$")
    plt.legend()
    plt.show()

    # 计算误差
    y_res = np.abs(y_true - y_predict)
    plt.plot(x, y_res, 'r-')
    plt.xlabel("x")
    plt.ylabel("res")
    plt.title("Residual between Math and GDA")
    plt.show()

# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS

"""
微分求解
scipy提供的方法与自己实现的方法对比
odeint()函数是scipy库中一个数值求解微分方程的函数
odeint()函数需要至少三个变量，第一个是微分方程函数，第二个是微分方程初值，第三个是微分的自变量。

测试一：
二元微分计算dy/dx = y - 2x/y
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt

from algorithm.differential.Runge.RungeMethod import RungeKutta
from algorithm.differential.test1Func import FuncImpl1


def diff(y, x):
    return np.array(y - 2 * x / y)


def main():
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 使用scipy提供的方法
    x1 = np.linspace(0, 1, 11)
    print(x1)
    y1 = odeint(diff, 1, x1)

    fig1 = plt.figure("使用scipy的odeint()计算效果图")
    plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.75)  # 调整子图间距

    plt.subplot(1, 2, 1)
    plt.plot(x1, y1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("odeint()方法")

    # 2. 手动实现的龙格库塔计算
    runge = RungeKutta(0, 1, 1, 0.1)
    func = FuncImpl1()
    yn, x2, y2 = runge.calculate(func)

    plt.subplot(1, 2, 2)
    plt.plot(x2, y2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Runge龙格库塔方法")
    plt.suptitle("微分求解方法对比")
    plt.show()

    # 3. 利用一张图对比
    plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.6, hspace=0.75)  # 调整子图间距
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, color='blue', alpha=0.5, lw=2, label="odeint")
    plt.plot(x2, y2, color='red', alpha=0.5, lw=2, label="runge")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("两种方法计算结果对比")
    plt.legend()

    # 4. 绘制误差图
    y1 = np.ravel(y1.reshape(1, -1))
    y2 = np.ravel(np.array(y2).reshape(1, -1))
    print("y1.shape : ", y1.shape)
    print("y2.shape : ", y2.shape)
    print("y1 : ", y1)
    print("y2 : ", y2)

    yref = np.ravel(np.abs(y1-y2).reshape(1, -1)[0])
    #
    print(yref.shape)
    print(yref)

    plt.subplot(1, 2, 2)
    plt.plot(x1, yref, color='green', lw=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("两种方法的偏差")
    plt.show()


if __name__ == "__main__":
    main()





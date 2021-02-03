# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

"""
多元RK34实现，含验证与绘图
目前实现的这个，返回结果不含初始值，绘图是从x+step的位置开始
"""


class MultiFunc(object):

    def getFunction(self, parameters):
        """
        算例1：捕食者被捕食者模型
        dx/dt = 2x - 0.02xy
        dy/dt = 0.0002xy - 0.8y
        初值：x = 3000, y = 120
        :param iniValues: 初始值
        :return: num数组
        """
        num = np.zeros_like(parameters)
        num[0] = parameters[0]
        num[1] = 2 * parameters[1] - 0.02 * parameters[1] * parameters[2]
        num[2] = 0.0002 * parameters[1] * parameters[2] - 0.8 * parameters[2]

        return num


class RungeKutta(object):

    def calculate(self, multiFunc, inivalues, xn, step):
        """
        计算多元RK34的方法
        :param multifunc: 多元计算函数
        :param inivalues: 初始值
        :param xn: 终止条件
        :param step: 步长
        :return:
        """
        length = inivalues.shape[0]
        yn = np.zeros_like(inivalues)
        y = inivalues.copy()
        x = inivalues[0]
        values = []

        while x < xn:
            k1 = multiFunc.getFunction(y)
            k2ParaValues = np.zeros_like(inivalues)
            k2ParaValues[0] = x + step / 2
            for i in range(1, length):
                k2ParaValues[i] = y[i] + k1[i] * step / 2

            k2 = multiFunc.getFunction(k2ParaValues)

            k3ParaValues = np.zeros_like(inivalues)
            k3ParaValues[0] = x + step / 2
            for i in range(1, length):
                k3ParaValues[i] = y[i] + k2[i] * step / 2

            k3 = multiFunc.getFunction(k3ParaValues)

            k4ParaValues = np.zeros_like(inivalues)
            k4ParaValues[0] = x + step
            for i in range(1, length):
                k4ParaValues[i] = y[i] + k3[i] * step

            k4 = multiFunc.getFunction(k4ParaValues)

            yn[0] = x

            ivalues = [x,]
            for i in range(1, length):
                yn[i] = y[i] + step * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
                ivalues.append(yn[i])

            values.append(ivalues)
            x += step
            y = yn.copy()
            y[0] = x

        return y, values

    def plotNum(self, values, msg):
        valueArray = np.array(values)
        t = valueArray[:, 0]
        x = valueArray[:, 1]
        y = valueArray[:, 2]


        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ymajorlocator1 = MultipleLocator(1000)  # 将1轴主刻度标签设置为1000的倍数
        ymajorlocator2 = MultipleLocator(20)  # 将y2轴主刻度标签设置为20的倍数
        ax1.yaxis.set_major_locator(ymajorlocator1)
        ax2.yaxis.set_major_locator(ymajorlocator2)

        line1,  = ax1.plot(t, x, color="b")
        line2, = ax2.plot(t, y, color="red")
        ax1.set_xlabel("t")
        ax1.set_ylabel("rabbit num", color="b")
        ax2.set_ylabel("fox num", color="red")
        ax1.set_ylim(2000, 6000)
        ax2.set_ylim(60, 140)
        plt.title(msg)
        plt.legend((line1, line2), ("rabbit", "fox"))
        plt.show()


if __name__ == "__main__":
    inivalues = np.array([0.0, 3000.0, 120.0])
    runge = RungeKutta()
    multiFunc = MultiFunc()
    y, values = runge.calculate(multiFunc, inivalues, 20, 0.1)
    runge.plotNum(values, "predator-prey model")

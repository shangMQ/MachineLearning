# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
import numpy as np


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


if __name__ == "__main__":
    inivalues = np.array([0.0, 300.0, 120.0])
    multiFunc = MultiFunc()
    y = multiFunc.getFunction(inivalues)
    print(y)
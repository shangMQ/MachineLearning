# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS

"""
    龙格库塔算法
    成员变量：
         x0:  x的初值
         xn:  x的终值
         y0:  y的初值
          h:   步长
"""


class RungeKutta(object):

    x0 = 0.0
    xn = 0.0
    y0 = 0.0
    h = 0.0

    def __init__(self, x0, xn, y0, h):
        self.x0 = x0
        self.xn = xn
        self.y0 = y0
        self.h = h

    def calculate(self, Func):

        x = self.x0
        y = self.y0

        xlist = [self.x0]
        ylist = [self.y0]

        while abs(self.xn - x) > 1.0e-5:
            k1 = Func.getFunction(x, y)
            k2 = Func.getFunction(x + self.h / 2, y + self.h * k1 / 2)
            k3 = Func.getFunction(x + self.h / 2, y + self.h * k2 / 2)
            k4 = Func.getFunction(x + self.h, y + self.h * k3)

            yn = y + self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            x += self.h
            y = yn
            xlist.append(x)
            ylist.append(y)

        return y, xlist, ylist



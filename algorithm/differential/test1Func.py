# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
from algorithm.differential.Runge.Func import Func


class FuncImpl1(Func):
    def getFunction(self, x, y):
        return y - 2 * x / y
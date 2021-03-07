# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
利用梯度下降算法计算一个数的平方根
"""

import math

if __name__ == "__main__":
    learning_rate = 0.01

    for num in range(1, 100):
        cur = 0

        for i in range(1000):
            cur -= learning_rate * (cur**2 - num)

        print("%d的平方根计算，近似值 = %f, 真实值 = %f" % (num, cur, math.sqrt(num)))
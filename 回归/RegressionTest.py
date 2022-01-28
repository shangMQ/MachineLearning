# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:53:20 2019
利用梯度下降法计算平方根
令g(x)的导数等于f(x)=x^2-a，当导数为零的时候，g(x)取得了极小值，此时x=sqrt(a)
@author: Kylin
"""
import math

if __name__ == "__main__":
    learning_rate = 0.01
    for a in range(1, 100):
        cur = 0
        for i in range(1000):
            # 令f(x)=0的最小的cur值，就是我们预测的平方根
            cur -= learning_rate*(cur**2 - a)
        print("{:}的平方根的真实值是{:.4f},预测值为{:.4f}".format(a, math.sqrt(a), cur))
    


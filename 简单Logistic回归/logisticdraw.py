# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:39:42 2019
绘制1/(1+e-x)
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 100)
y = 1 / (1+np.exp(-x))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = 1 / (1 + exp(-x))")


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:45:41 2019
损失函数：logistic损失、hinge损失及0/1损失
@author: Kylin
"""
import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.array(np.linspace(-3,3,1001, dtype=np.float))
    y_logit = np.log(1 + np.exp(-x))/np.log(2)
    y_01 = x < 0
    
    #合页损失函数，当分类正确时，1-x<0,损失为0；当分类错误时，损失为1-x
    y_hinge = 1 - x
    y_hinge[y_hinge < 0] = 0
    plt.plot(x, y_logit, 'r--', label = 'logistic loss', linewidth=2)
    plt.plot(x, y_01, 'b-', label = '0/1 loss', linewidth=2)
    plt.plot(x, y_hinge, 'g-', label = 'SVM loss', linewidth=2)
    plt.grid()
    plt.title("Compared with Loss Function")
    plt.legend()
    plt.show()

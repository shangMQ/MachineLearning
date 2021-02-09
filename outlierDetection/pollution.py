# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
水质数据的异常检测
"""

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


def showData(data):
    plt.figure(figsize=(10, 6), facecolor='w')

    x = data['H2O'].values
    plt.plot(x, 'r-', lw=1, label='C0911')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title('RealData0911', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(b=True, ls=':', color='#404040')
    plt.show()


if __name__ == "__main__":
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    # 1. 读取数据
    data = pd.read_csv('C0911.csv', header=0)   # C0911.csv
    print(data.shape)
    print(data.head())
    # showData(data)

    # 获取水质数据
    x = data['H2O'].values
    print(x)

    # 异常检测
    width = 500
    delta = 10
    eps = 0.15
    N = len(x)
    p = []
    abnormal = []
    for i in np.arange(0, N-width, delta):
        s = x[i:i+width]
        # np.ptp()计算数组中最大值与最小值的差
        p.append(np.ptp(s))
        if np.ptp(s) > eps:
            abnormal.append(list(range(i, i+width)))
    abnormal = np.unique(abnormal)
    print("Abnormal Time:\n", abnormal)
    print("Abnormal Time num:", len(abnormal))

    # 绘制异常间隔数据
    plt.figure(facecolor='w')
    plt.plot(p, lw=1)
    plt.grid(b=True, ls=':', color='#404040')
    plt.title('Fixed interval difference', fontsize=16)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Difference', fontsize=15)
    plt.show()

    # 绘制真实数据
    plt.figure(figsize=(11, 5), facecolor='w')
    plt.subplot(131)
    plt.plot(x, 'r-', lw=1, label='Initial Data')
    plt.title('Real Data', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(b=True, ls=':', color='#404040')

    plt.subplot(132)
    t = np.arange(N)
    plt.plot(t, x, 'r-', lw=1, label='initial data')
    plt.plot(abnormal, x[abnormal], 'go', markeredgecolor='g', ms=2, label='outlier')
    plt.legend(loc='upper right')
    plt.title('Outlier detection', fontsize=16)
    plt.grid(b=True, ls=':', color='#404040')

    # 异常校正(预测)
    plt.subplot(133)
    select = np.ones(N, dtype=np.bool)
    select[abnormal] = False
    print("select shape : ", select.shape)
    t = np.arange(N)
    # 使用基于决策树的集成学习模型对数据进行预测
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
    br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
    br.fit(t[select].reshape(-1, 1), x[select])
    y = br.predict(np.arange(N).reshape(-1, 1))
    y[select] = x[select]

    plt.plot(x, 'g--', lw=1, label='initial')    # 原始值
    plt.plot(y, 'r-', lw=1, label='correction')     # 校正值
    plt.legend(loc='upper right')
    plt.title('Outlier correction', fontsize=16)
    plt.grid(b=True, ls=':', color='#404040')

    plt.tight_layout(1.5, rect=(0, 0, 1, 0.95))
    plt.suptitle('Outlier detection and correction', fontsize=18)
    plt.show()

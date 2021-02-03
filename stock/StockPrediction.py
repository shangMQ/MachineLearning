# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
股票预测：
利用简单滑动平均和指数滑动平均
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 1. 读取数据
    stock_max, stock_min, stock_close, stock_amount = np.loadtxt('SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True, encoding='gb18030')

    # 由于数据有点多，只打印前100个的收盘价格
    N = 100
    stock_close = stock_close[:N]
    print(stock_close)

    # 2. 简单移动平均，利用卷积
    n = 5
    weight = np.ones(n)
    weight /= weight.sum()
    print("简单移动平均的权重：", weight)
    stock_sma = np.convolve(stock_close, weight, mode='valid')  # simple moving average

    # 3. 指数移动平均
    weight = np.linspace(1, 0, n)
    print(weight)
    weight = np.exp(weight)
    weight /= weight.sum()
    print("指数移动平均的权重：", weight)
    stock_ema = np.convolve(stock_close, weight, mode='valid')  # exponential moving average

    # 4. 利用多项式拟合
    t = np.arange(n-1, N)
    poly = np.polyfit(t, stock_ema, 5)
    print(poly)
    t = np.arange(n-1, N)
    stock_ema_hat = np.polyval(poly, t)

    mpl.rcParams['font.sans-serif'] = [u'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label='原始收盘价', mec='k')
    t = np.arange(n-1, N)
    plt.plot(t, stock_sma, 'b-', linewidth=2, label='简单移动平均线')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label='指数移动平均线')
    plt.legend(loc='upper right')
    plt.title('股票收盘价与滑动平均线MA', fontsize=15)
    plt.grid(b=True, ls=':', color='#404040')
    plt.show()

    print(plt.figure(figsize=(7, 5), facecolor='w'))
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=1, label='原始收盘价', mec='k')
    plt.plot(t, stock_ema_hat, '-', color='#4040FF', linewidth=3, label='指数移动平均线估计')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label='指数移动平均线')
    plt.legend(loc='upper right')
    plt.title('滑动平均线MA的估计', fontsize=15)
    plt.grid(b=True, ls=':', color='#404040')
    plt.show()

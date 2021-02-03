# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
股票K线绘制
使用mplfinance库
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据
    np.set_printoptions(suppress=True, linewidth=100, edgeitems=5)
    data = pd.read_csv("SH600000.csv", header=1)
    print(data)

    N, m = data.shape
    print(N)

    # 2. 修改列名
    data_column = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'n']
    data.columns = data_column

    # 3. 对时间序列进行处理
    print("原始数据：")
    print(data.head())
    print(data.describe())
    print("日期:")
    print(data.trade_date.head())

    # 由于目前的日期是字符串，先对这个进行处理,并转化为datetime类型的对象
    data['datetime'] = pd.to_datetime(data.trade_date)
    data = data.set_index('datetime')
    data = data.drop(['trade_date'], axis=1)
    print(data.head())

    # 4. 使用方法
    # 基础绘制
    # mpf.plot(data)
    # mpf.plot(data, type='candle')

    # 绘制均线
    # mpf.plot(data, type='candle', mav=(2, 5, 10), style='charles')

    # 绘制成交量
    mpf.plot(data, type='candle', mav=(2, 5, 10), style='charles', volume=True)

    # t = np.arange(1, N+1).reshape((-1, 1))
    # data = np.hstack((t, data))
    #
    # fig, ax = plt.subplots(facecolor='w')
    # fig.subplots_adjust(bottom=0.2)
    # mpf.candlestick_ohlc(ax, data, width=0.6, colorup='r', colordown='g', alpha=0.9)
    # plt.xlim((0, N+1))
    # plt.grid(b=True, ls=':', color='#404040')
    # plt.title('股票K线图', fontsize=15)
    # plt.tight_layout(2)
    # plt.show()

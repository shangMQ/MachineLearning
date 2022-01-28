# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning

"""
ARIMA自回归移动平均模型的应用
p:Autoregressive Regression自回归项，预测模型中采用的时序滞后数
  · 描述当前值与历史值之间的关系，用变量自身的历史时间数据对自身进行预测
  · 自回归模型必须满足平稳性要求
  · 必须具有自相关性
  · 自回归只适用于预测与自身前期相关的现象
d:数据进行几阶差分化，才是稳定的
q:Moving Average移动平均模型， 关注误差
  1. 自回归模型中的误差项的累加
  2. 消除预测中的随机波动
"""


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


def date_parser(date):
    return pd.datetime.strptime(date, '%Y-%m')


if __name__ == '__main__':
    # 相关设定
    warnings.filterwarnings(action='ignore', category=HessianInversionWarning)
    pd.set_option('display.width', 100)
    np.set_printoptions(linewidth=100, suppress=True)
    mpl.rcParams['font.sans-serif'] = u'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据
    data = pd.read_csv('AirPassengers.csv', header=0, parse_dates=['Month'], date_parser=date_parser, index_col=['Month'])
    data.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
    print("数据类型：")
    print(data.dtypes)

    # 发现通过pandas直接读入的数值为object类型，为了便于操作，将其转化为float类型
    x = data['Passengers'].astype(np.float)
    x = np.log(x)
    print("log(x):")
    print(x.head(10))

    # 2. 对数据进行相关处理，即进行差分处理
    # show = 'prime'   # 'diff', 'ma', 'prime'
    show = 'prime'
    d = 1

    # np.shift()前一个减去后一个，第一个值为NAN
    print(x.shape)
    diff = x - x.shift(periods=d)

    # pd.diff()也可以计算出差分值

    # np.rolling()移动窗口
    # 参数1：window：表示时间窗的大小，注意有两种形式（int or offset）
    #       如果使用int，则数值表示计算统计量的观测值的数量即向前几个数据。
    #       如果是offset类型，表示时间窗的大小
    # 参数2：min_periods：最少需要有值的观测点的数量，对于int类型，默认与window相等
    # 参数3：center：是否使用window的中间值作为label，默认为false。
    ma = x.rolling(window=12).mean()
    xma = x - ma

    # 3. 利用ARIMA自回归移动平均模型进行分析
    p = 8
    q = 8
    model = ARIMA(endog=x, order=(p, d, q))     # 自回归函数p,差分d,移动平均数q
    arima = model.fit(disp=-1)                  # disp<0:不输出过程
    prediction = arima.fittedvalues
    print(type(prediction))
    y = prediction.cumsum() + x[0]
    mse = ((x - y)**2).mean()
    rmse = np.sqrt(mse)

    # 3. 可视化操作
    plt.figure(facecolor='w')
    if show == 'diff':
        # 1阶差分可视化
        plt.plot(x, 'r-', lw=2, label='Initial')
        plt.plot(diff, 'g-', lw=2, label='%d order difference' % d)
        plt.plot(prediction, 'b-', lw=2, label=u'predict')
        title = 'Log(Passenger Num)'
    elif show == 'ma':
        # 滑动平均值与MA预测值
        plt.plot(x, 'r-', lw=2, label=u'ln(Initial)')  # 原始数据的对数值
        plt.plot(ma, 'g-', lw=2, label=u'average mean')  # 移动平均值
        plt.plot(xma, 'b-', lw=2, label='ln(Initial) - ln(MA))')
        plt.plot(prediction, 'k-', lw=2, label='Predict')
        title = 'Average of Moving and MA prediction'
    else:
        # 查看预测结果
        plt.plot(x, 'r-', lw=2, label='Initial')
        plt.plot(y, 'g-', lw=2, label='Predict')
        title = 'Prediction Result(AR=%d, d=%d, MA=%d):RMSE=%.4f' % (p, d, q, rmse)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.title(title, fontsize=16)
    plt.tight_layout(2)
    # plt.savefig('%s.png' % title)
    plt.show()

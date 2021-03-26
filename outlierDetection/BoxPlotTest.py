# -- coding = 'utf-8' --
# Author Kylin
# Python Version 3.7.3
# OS macOS

"""
基于水质数据的箱型图绘制，用于检测异常值
第一四分位数（Q1） 样本所有数值由小到大排列后第25%的数字
中位数（Q2） 样本所有数值由小到大排列后第50%的数字
第三四分位数(Q3) 样本所有数值由小到大排列后第75%的数字
异常值：大于Q3+1.5IQR的值，或者小于Q3-1.5IQR
"""

import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt


def showData(data, text):
    plt.figure(figsize=(10, 6), facecolor='w')

    x = data['OutletValve'].values
    plt.plot(x, 'r-', lw=1, label='C0911')
    plt.xlabel("Time")
    plt.ylabel(text)
    plt.title('RealData0911', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(b=True, ls=':', color='#404040')
    plt.show()


if __name__ == "__main__":
    # 字体设置
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    # 1. 读取数据
    data = pd.read_csv('C0911.csv', header=0)   # C0911.csv
    print(data.shape)
    print(data.head())
    showData(data, 'OutletValve')

    # 获取水质数据
    x = data['OutletValve'].values
    print(x)

    # 2. 绘制箱型图
    plt.style.use('ggplot')
    plt.boxplot(x,  # 选择数据
                patch_artist=True,  # 上下四分位是否填充
                showmeans=True,  # 以点的形式显示均值
                boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色
                meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色
                medianprops={'linestyle': '--', 'color': 'orange'},  # 设置中位数线的属性，线的类型和颜色
                )

    # 去除箱线图的上边框与右边框的刻度标签
    plt.tick_params(top='off', right='off')

    # 显示图形
    plt.show()



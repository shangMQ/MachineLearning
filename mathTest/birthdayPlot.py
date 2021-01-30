# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
计算至少两人是同年同月同日生的概率
"""
import matplotlib.pyplot as plt
import matplotlib as mpl


def Ann(n, N):
    """
    手动计算排列数AnN
    :param n:
    :param N:
    :return:
    """
    sum = 1
    for i in range((N-n+1), N+1):
        sum *= i
    return sum


def calculate(n):
    """
    计算概率
    :param n:
    :return:
    """
    return 1 - Ann(n, 365) / 365 ** n


def plotProbability(plist):
    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.plot(plist, 'g-')
    plt.xlabel("person num", fontsize=16)
    plt.ylabel("probability", fontsize=16)
    plt.title("Birthday test", fontsize=18)
    plt.show()


def main():
    num = int(input("请输入人数的范围："))
    plist = []

    # 1. 计算各种情况下的概率
    for i in range(1, num+1):
        plist.append(calculate(i))

    # 2. 概率绘制
    plotProbability(plist)


if __name__ == "__main__":
    main()
# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
Ｎagel-Schreckenberg交通流模型——公路堵车概率模型
路面上有n辆车，以不同的速度向前行驶，模拟堵车问题。
    假设：
    1.假设某辆车的当前速度是v。
    2.若前方可见范围内没车，则它在下一秒的车速提高到(v+1)，直到达到规定的最高限速。
    3.若前方有车，前车的距离为d，且d < v，则它下一秒的车速降低到(d-1) 。
    4.每辆车会以概率p随机减速(v-1)。
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def clip(x, path):
    """
    对车辆的位移进行限制，当大于水平的公路长度时，则认为是拐外后的位移
    :param x:
    :param path:
    :return:
    """
    for i in range(len(x)):
        if x[i] >= path:
            x[i] %= path


def main():
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    path = 5000  # 环形公路的长度
    n = 100  # 公路中的车辆数目
    v0 = 5  # 车辆的初始速度
    p = 0.3  # 随机减速概率
    Times = 3000  # 模拟时间

    # 1. 模拟车辆的位移和速度
    np.random.seed(0)
    # 模拟100辆车辆的位置
    x = np.random.rand(n) * path
    # 排序
    x.sort()

    v = np.tile([v0], n).astype(np.float)

    plt.figure(figsize=(10, 8), facecolor='w')

    for t in range(Times):
        # 绘图，其中x表示车辆此时的位置，[t*n]表示时间
        plt.scatter(x, [t]*n, s=1, c='b', alpha=0.05)
        # 依次判断这100辆车的速度和位移情况
        for i in range(n):
            # 计算前后车辆距离
            if x[(i+1) % n] > x[i]:
                # 如果前车没有拐弯
                d = x[(i+1) % n] - x[i]
            else:
                # 环形位置点重置
                d = path - x[i] + x[(i+1) % n]

            # 判断此刻的车速与前车距离
            if v[i] < d:
                # 若前方可见范围内没有车，可将车速提升
                if np.random.rand() > p:
                    v[i] += 1
                else:
                    # 随机减速
                    v[i] -= 1
            else:
                # 若前方有车，则减速
                v[i] = d - 1
        # 限制速度，v < 0，则将v定义为0；v>150，则将v定义为150
        v = v.clip(0, 150)
        # 车辆位移增加
        x += v
        clip(x, path)

    plt.xlim(0, path)
    plt.ylim(0, Times)

    # 标签,显示设置字体
    plt.xlabel('Vehicle Location', fontsize=16)
    plt.ylabel('Times', fontsize=16)
    plt.title(u'Nagel-Schreckenberg Traffic Stream Simulation with random reduce speed ratio of %.2f' % p, fontsize=16)

    '''自动调整子图参数，使之填充整个图像区域'''
    plt.tight_layout(pad=2)

    '''画图'''
    plt.show()


if __name__ == "__main__":
    main()


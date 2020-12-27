# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
多维高斯分布
"""
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def main():
    # 1. 生成数据
    x1, x2 = np.mgrid[-5:5:51j, -5:5:51j]
    x = np.stack((x1, x2), axis=2)
    print('x1 = \n', x1)
    print("X1.shape : ", x1.shape)
    print('x2 = \n', x2)
    print("X2.shape : ", x2.shape)
    print('x = \n', x)

    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    plt.figure(figsize=(9, 8), facecolor='w')

    # 4组不同的sigma值
    sigma = (np.identity(2), np.diag((3, 3)), np.diag((2, 5)), np.array(((2, 1), (1, 5))))
    print(sigma)
    # 绘制4组图形
    for i in np.arange(4):
        ax = plt.subplot(2, 2, i + 1, projection='3d')
        norm = stats.multivariate_normal((0, 0), sigma[i])
        y = norm.pdf(x)
        ax.plot_surface(x1, x2, y, cmap=cm.Accent, rstride=1, cstride=1, alpha=0.9, lw=0.3, edgecolor='#303030')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.suptitle('Gauss Distribution with two dimension', fontsize=18)
    plt.tight_layout(1.5)
    plt.show()


if __name__ == '__main__':
    main()
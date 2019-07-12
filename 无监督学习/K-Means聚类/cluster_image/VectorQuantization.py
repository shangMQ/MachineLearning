# !/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image #PIL库是python image library库
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def restore_image(cb, cluster, shape):
    """
    存储图像
    参数：
        cb:聚类质心
        cluster：聚类结果
        shape:图像形状（512，512，3）
    返回值：
        利用聚类信息生成的图像
    """
    row, col, dummy = shape
    image = np.empty((row, col, 3))
    index = 0
    for r in range(row):
        for c in range(col):
            image[r, c] = cb[cluster[index]]
            index += 1
    return image


def show_scatter(a):
    """
    显示关于数组a的三维散点图
    参数：
        图像数组a
    """
    N = 10
    print('原始数据：\n', a)
    density, edges = np.histogramdd(a, bins=[N,N,N], range=[(0,1), (0,1), (0,1)])
    density /= density.max()
    x = y = z = np.arange(N)
    d = np.meshgrid(x, y, z)

    fig = plt.figure("图像颜色三维散点图", facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d[1], d[0], d[2], c='r', s=100*density, marker='o', depthshade=True)
    ax.set_xlabel(u'红色分量')
    ax.set_ylabel(u'绿色分量')
    ax.set_zlabel(u'蓝色分量')
    plt.title(u'图像颜色三维频数分布', fontsize=20)

    plt.figure("图像颜色频数分布图", facecolor='w')
    den = density[density > 0]
    den = np.sort(den)[::-1]
    t = np.arange(len(den))
    plt.plot(t, den, 'r-', t, den, 'go', lw=2)
    plt.xlabel("颜色个数")
    plt.ylabel("颜色频数")
    plt.title(u'图像颜色频数分布', fontsize=18)
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    #设置中文字体
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    #使用颜色个数
    num_vq = 50
    
    #打开图像
    im = Image.open('lena.png')     # son.bmp(100)/flower2.png(200)/son.png(60)/lena.png(50)
    image = np.array(im).astype(np.float) / 255 #对图像做均一化,im转化为array数组之后是512行512列，3个rgb通道
    image = image[:, :, :3]
    image_v = image.reshape((-1, 3))#全都转化为3个通道为一行的数组
    model = KMeans(num_vq)#使用我们想要的颜色总数作为聚类数
    show_scatter(image_v)#显示关于图像的三维散点图

    N = image_v.shape[0]    # 图像像素总数
    # 选择足够多的样本(如1000个)，计算聚类中心
    idx = np.random.randint(0, N, size=1000)
    image_sample = image_v[idx]
    model.fit(image_sample)
    c = model.predict(image_v)  # 聚类结果
    print('聚类结果：\n', c)
    print('聚类中心：\n', model.cluster_centers_)

    plt.figure(figsize=(15, 8), facecolor='w')
    plt.subplot(121)
    plt.axis('off')
    plt.title(u'原始图片', fontsize=18)
    plt.imshow(image)

    plt.subplot(122)
    vq_image = restore_image(model.cluster_centers_, c, image.shape)
    plt.axis('off')
    plt.title(u'矢量量化后图片：%d色' % num_vq, fontsize=18)
    plt.imshow(vq_image)
    plt.savefig('cluster_lena.png')

    plt.tight_layout(1.2)
    plt.show()

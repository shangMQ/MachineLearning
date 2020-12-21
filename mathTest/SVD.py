# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
SVD奇异值分解——图像实例
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from PIL import Image


def restore(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    # a[a < 0] = 0
    # a[a > 255] = 255
    # np.clip()是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
    a = a.clip(0, 255)
    # np.rint()是根据四舍五入取整
    return np.rint(a).astype('uint8')


def main():
    mpl.rcParams["font.sans-serif"] = [u'Times New Roman']
    mpl.rcParams[u'axes.unicode_minus'] = False

    # 1. 读取一个图片
    A = Image.open("cat.jpg", 'r')

    # 2. 创建图片的输出路径，如果不存在就创建一个
    output_path = r'../mathTest/pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 3. 将图片转换为数组(行，列，3通道)
    a = np.array(A)
    # print(a.shape)
    # print(a)

    # # 4. 使用SVD分解
    K = 20
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])

    plt.figure(figsize=(10, 10), facecolor='w')

    for k in range(1, K+1):
        print(k)
        R = restore(sigma_r, u_r, v_r, k)
        G = restore(sigma_g, u_g, v_g, k)
        B = restore(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), 2)
        Image.fromarray(I).save('%s/svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')  # 关闭坐标轴
            plt.title("Singular Value Num %d" % k)

    plt.suptitle("Image deposition by SVD")
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    main()
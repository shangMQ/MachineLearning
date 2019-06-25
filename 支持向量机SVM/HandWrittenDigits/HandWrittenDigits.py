#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from PIL import Image
import os


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + '正确率：%.2f%%' % (100*np.mean(acc)))


def save_image(im, i):
    """
    保存图像（不太懂这里的逻辑）
    """
    im *= 15.9375
    im = 255 - im
    a = im.astype(np.uint8)
    output_path = '.\\HandWritten'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Image.fromarray(a).save(output_path + ('\\%d.png' % i))


if __name__ == "__main__":
    
    print('Load Training File Start...')
    #加载训练数据
    data = np.loadtxt('optdigits.tra', dtype=np.float, delimiter=',')
    x, y = np.split(data, (-1, ), axis=1)
    images = x.reshape(-1, 8, 8)
    y = y.ravel().astype(np.int)

    print('Load Test Data Start...')
    #加载测试数据
    data = np.loadtxt('optdigits.tes', dtype=np.float, delimiter=',')
    x_test, y_test = np.split(data, (-1, ), axis=1)
    images_test = x_test.reshape(-1, 8, 8)
    y_test = y_test.ravel().astype(np.int)
    print('Load Data OK...')

    # x, x_test, y, y_test = train_test_split(x, y, random_state=1)
    # images = x.reshape(-1, 8, 8)
    # images_test = x_test.reshape(-1, 8, 8)

    #设置中文字体
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    #显示灰度图像（前16个是训练数据的图像，后16个事测试数据的图像）
    plt.figure(figsize=(15, 9), facecolor='w')
    for index, image in enumerate(images[:16]):
        plt.subplot(4, 8, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练图片: %i' % y[index])
    for index, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, index + 17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(image.copy(), index)
        plt.title(u'测试图片: %i' % y_test[index])
    plt.tight_layout()
    plt.show()

    
    #利用训练数据拟合svm模型
    clf = svm.SVC(C=1, kernel='rbf', gamma=0.001)   # gamma很小时，近似于K近邻~ kNN
    print('Start Learning...')
    clf.fit(x, y)
    print('Learning is OK...')
    y_hat = clf.predict(x)
    show_accuracy(y, y_hat, '训练集')
    y_hat = clf.predict(x_test)
    print(y_hat)
    print(y_test)
    print("测试集正确率：", accuracy_score(y_test, y_hat))
    
    #拿出分错的图片
    err_images = images_test[y_test != y_hat]
    err_y_hat = y_hat[y_test != y_hat]
    err_y = y_test[y_test != y_hat]
    print("分类错误的图片：")
    print("预测值：", err_y_hat)
    print("真实值：", err_y)
    
    #显示分错的前12个图像
    plt.figure(figsize=(10, 8), facecolor='w')
    for index, image in enumerate(err_images):
        if index >= 12:
            break
        plt.subplot(3, 4, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("错分为：%i，真实值：%i".format(err_y_hat[index], err_y[index]))
    plt.tight_layout()
    plt.show()

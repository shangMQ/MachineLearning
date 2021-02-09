# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
词云图简化绘制，词的字体大小体现了词的出现频率
"""

import numpy as np
import collections
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import jieba
import re


def collection_words(file_path):
    texts = []
    time_pattern = re.compile(r'\d{1,2}:\d{1,2}:\d{1,2}')
    f = open(file_path, mode='r', encoding='gbk')

    for line in f:
        line = line.strip()

        if line:
            t = re.findall(pattern=time_pattern, string=line)
            if len(t) >= 1:
                continue
            texts += jieba.cut(line)

    f.close()
    return texts


def rand_color(word, font_size, position, orientation, random_state, font_path):
    hues = (0, 60, 120, 180, 240, 300)  # 0°红、60°黄、120°绿、180°青、240°蓝、300°洋红
    hue = hues[np.random.randint(len(hues))] # r∈[low, high)
    saturation = np.random.randint(50, 100)
    lightness = np.random.randint(0, 100)
    return 'HSL(%d, %d%%, %d%%)' % (hue, saturation, lightness)


def convolve(image, weight):
    height, width = image.shape
    h, w = weight.shape
    height_new = height - h + 1
    width_new = width - w + 1
    image_new = np.zeros((height_new, width_new), dtype=np.float)
    for i in range(height_new):
        for j in range(width_new):
            image_new[i,j] = np.sum(image[i:i+h, j:j+w] * weight)
    image_new = image_new.clip(0, 255)
    image_new = np.rint(image_new).astype('uint8')

    image_last = np.zeros((height, width), dtype=np.uint8)
    image_last[h//2:height-h//2, w//2:width-h//2] = image_new
    return image_last


def outline(file_name):
    im = Image.open(file_name)
    image = np.array(im).astype(np.float)
    soble = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
    R = convolve(image[:, :, 0], soble)
    G = convolve(image[:, :, 1], soble)
    B = convolve(image[:, :, 2], soble)
    I = np.stack((R, G, B), 2)
    return I, image.astype(np.int)


if __name__ == '__main__':
    # 1. 读取图片
    image_file = 'back.jpg'  # Lady.bmp
    image, mask = outline(image_file)
    imageData = Image.open('back.jpg')  # 作为背景形状的图
    graph = np.array(imageData)

    # 2. 读取文本
    texts = collection_words('story.txt')
    print("texts:\n", texts)
    # 高性能容量数据类型,查看单个词出现的次数
    cc = collections.Counter(texts)
    print("词汇出现次数:\n", cc)
    # 格式化词频，词频以字典形式存储
    cc = dict(zip(list(cc.keys()), list(cc.values())))
    print("格式化结果：\n", cc)

    max_font_size = 100
    wc = WordCloud(font_path='/System/Library/Fonts/PingFang.ttc', mask=mask,
                   background_color='black', max_font_size=max_font_size, max_words=len(cc))

    # 根据给定词频生成词云
    wc.generate_from_frequencies(cc)

    image_color = ImageColorGenerator(graph)
    plt.imshow(wc)
    plt.axis('off')  # 注意：不要显示坐标轴
    plt.tight_layout(1)
    plt.show()

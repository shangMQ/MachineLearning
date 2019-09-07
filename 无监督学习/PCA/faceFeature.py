# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:03:35 2019
对于面部识别的特征提取
@author: Kylin
"""
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    """
    加载数据
    """
    faces = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    
    #查看每张图片的大小
    face_shape = faces.images[0].shape
    print("每张图片的像素大小:", face_shape)
    
    #查看图片总个数
    print("faces.images.shape:", faces.images.shape)
    
    #查看类别总数
    print("图片来自{}位名人".format(len(faces.target_names)))
    
    return faces

def showFacesPic(faces):
    """
    查看前十张图片
    """
    fig, axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(), 'yticks':()})
    
    for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(faces.target_names[target])


def celebrityCount(faces):
    """
    查看名人目标图像的个数
    """
    print("The number of pictures with each celebrity:")
    counts = np.bincount(faces.target)
    for i, (count, name) in enumerate(zip(counts, faces.target_names)):
        print("({},{})".format(name, count))


if __name__ == "__main__":
    #1. 加载数据
    faces = loadData()
    
    #2. 查看前十张脸部图片
    showFacesPic(faces)
    
    #3. 查看每个名人的图片数目
    celebrityCount(faces)
    #可以发现数据偏斜，Bush的图片数就很多。
    
    #4. 此时考虑降低数据偏斜，否则特征提取会被数量多的目标影响
    
# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
图片转化为字符文本
"""
import numpy as np
from PIL import Image
import cv2


def main():
    # 1. 读入图像
    # 两种读入图像的方式
    # 第一种，使用openCV
    # data = cv2.imread("pic/buting.jpg") # 可以用opencv打开图像
    # print(data)
    # print("数据类型：", type(data))
    # print("数据大小：", data.shape)

    # 第二种，使用PIL的Image模块
    image_file = "pic/buting.jpg"
    height = 100
    img = Image.open(image_file)
    print("原始数据类型：", type(img))
    data = np.array(img)
    print(data)
    print("数据类型：", type(data))
    print("数据大小：", data.shape)
    img_width, img_height = img.size

    width = int(1.8 * height * img_width // img_height)

    img = img.resize((width, height), Image.ANTIALIAS)
    pixels = np.array(img.convert('L'))
    print("读取的像素大小：", pixels.shape)

    chars = "MNHQ$OC?7>!:-;. "
    N = len(chars)
    step = 256 // N
    print(N)
    result = ''

    for i in range(height):
        for j in range(width):
            result += chars[pixels[i][j] // step]
        result += '\n'

    with open("pic/buting.txt", mode='w') as f:
        f.write(result)

    resultArray = np.array(result)
    print(resultArray)


if __name__ == "__main__":
    main()

# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
卷积（emm）没写完
"""
import numpy as np
from PIL import Image


def convolve(image, weight):
    height, width = image.shape
    h, w = weight.shape
    height_new = height - h + 1
    width_new = width - w + 1
    image_new = np.zeros((height_new, width_new), dtype=np.float)

    for i in range(height_new):
        for j in range(width_new):
            image_new[i, j] = np.sum(image[i:i+h, j:j+w] * weight)

    image_new = image_new.clip(0, 255)
    image_new = np.rint(image_new).astype('uint8')
    return image_new


def main():
    # 1. 读入图像
    image_file = "pic/lena.jpg"
    height = 100

    # 2. 打开图像
    img = Image.open(image_file)
    img_width, img_height = img.size

    width = int(1.8 * height * img_width // img_height)

    img = img.resize((width, height), Image.ANTIALIAS)
    pixels = np.array(img.convert('L'))
    print(pixels.shape)
    chars = "smq>?!:;."
    N = len(chars)
    step = 256 // N
    print(N)
    result = ''

    for i in range(height):
        for j in range(width):
            result += chars[pixels[i][j] // step]
        result += '\n'

    with open("pic/text.txt", mode='w') as f:
        f.write(result)


if __name__ == "__main__":
    main()
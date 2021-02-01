# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
数据取整
np.ceil()：向上取整
np.floor()：向下取整
np.rint(): 四舍五入
"""
import numpy as np


def main():
    data = np.array([1, 1.1, 1.5, 2.4])
    print("Initial data:")
    print(data)

    # 使用rint()取整
    data1 = np.rint(data)
    print("四舍五入取整：")
    print(data1)

    # 使用ceil()取整
    data2 = np.ceil(data)
    print("向上取整：")
    print(data2)

    # 使用floor()取整
    data3 = np.floor(data)
    print("向下取整：")
    print(data3)


if __name__ == "__main__":
    main()
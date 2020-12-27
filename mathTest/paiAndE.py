# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
近似计算pai与e
"""
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. 近似求解pai
    n = np.sum(1 / np.arange(1, 10000) ** 2)
    pai = np.sqrt(n * 6)
    print("------The value of pai-------")
    print("计算值 = {:.10f}, 真实值 = {:.10f}".format(pai, np.pi))

    # 2. 近似求解e
    n = np.arange(1, 20)
    e_value = np.sum(1 / n.cumprod()) + 1

    print("------The value of e-------")
    print("计算值 = {:.10f}, 真实值 = {:.10f}".format(e_value, np.e))


if __name__ == "__main__":
    main()
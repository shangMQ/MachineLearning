# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
Logistic损失函数变化
假设：
    1为正类，即分类正确
"""
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = np.linspace(0, 1, 100)
    J1 = - np.log2(p)
    J2 = - np.log2(1-p)

    plt.plot(p, J1, "r--", label="y = 1")
    plt.plot(p, J2, "g-", label="y = 0")
    plt.xlabel("p")
    plt.ylabel("J(w)")
    plt.title("Loss function of Logistic Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
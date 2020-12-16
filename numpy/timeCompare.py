# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
numpy与python数学库的时间比较

附加：程序运行时间记录
time.clock()在python3.8以后废除了
time()精度上相对没有那么高，而且受系统的影响，适合表示日期时间或者大程序的计时。
perf_counter()适合小一点的程序测试，会计算sleep()时间。
process_counter()适合小一点的程序测试，不会计算sleep()时间。
    此外Python3.7开始还提供了以上三个方法精确到纳秒的计时。分别是：
    time.perf_counter_ns()
    time.process_time_ns()
    time.time_ns()
    注意这三个精确到纳秒的方法返回的是整数类型。
"""
import numpy as np
import time
import math


def main():
    num = 10000
    # 1. numpy包进行计算
    x = np.linspace(0, 10, num)
    # np_start = time.process_time()
    np_start = time.time_ns()
    y = np.sin(x)
    np_time = time.time_ns() - np_start

    # 2. python数学库
    x = x.tolist()
    # math_start = time.process_time()
    math_start = time.time_ns()
    for i, n in enumerate(x):
        x[i] = math.sin(n)
    math_time = time.time_ns() - math_start

    print("numpy的运行时间：%d ns" % np_time)
    print("math的运行时间：%d ns" % math_time)
    print("math/numpy = %.2f" % (math_time / np_time))


if __name__ == "__main__":
    main()
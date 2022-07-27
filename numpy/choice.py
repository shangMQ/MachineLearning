"""
choice()可以从给定的列表中，以一定概率和方式抽取结果，当不指定概率时为均匀采样，默认抽取方式为有放回抽样：
permutation()全排列
"""
import numpy as np

my_list = ['a', 'b', 'c', 'd']
np.random.seed(0) # 对于随机数相关任务，最好在一开始使用种子，确保每次运行的结果一样
choiced = np.random.choice(my_list, 2, replace=False, p=[0.1, 0.7, 0.1 ,0.1])
print(choiced)

# 当返回的元素个数与原列表相同时，不放回抽样等价于使用 permutation 函数，即打散原列表：
permutated = np.random.permutation(my_list)
print(permutated)
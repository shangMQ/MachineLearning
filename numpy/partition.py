"""
np.partition()的用法
类似快排的算法，以第'k-th'位置为基准，该位置的元素位于对原始数组排序之后的位置，
大于该元素值的元素被放置在该元素的后面，小于该元素值的元素被放置在该元素的前面，
前后两端的元素排列顺序无要求。
参数：
kth:要分区的元素索引。元素的第k个值将处于其最终排序位置，所有较小的元素将移到它之前，所有相等或更大的元素都将移到它后面。 分区中所有元素的顺序未定义。
axis:要排序的轴。如果为 None，则数组在排序前被展平。默认值为 -1，沿最后一个轴排序。
"""
import numpy as np

a = np.array([[3, 4, 2, 1], [1, 5, 6, 3]])
print("---initial data---")
print(a)
print("---partition data by column---")
print(np.partition(a, kth=1, axis=0))
print("---partition data by row---")
print(np.partition(a, kth=1, axis=1))

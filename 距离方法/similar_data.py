import time
import torch
from torch import linalg as LA
from line_profiler import LineProfiler

"""
基于pytorch的最小距离计算
"""

def get_elu_dis(tensor_1, tensor_2):
    # 平方计算
    tensor_1_square = torch.pow(tensor_1, 2).sum(dim=0)
    tensor_2_square = torch.pow(tensor_2, 2).sum(dim=1)

    # 向量矩阵相乘
    multiply_sum = torch.sum(tensor_1 * tensor_2, dim=1)

    # 欧式距离计算
    euclidean_distance = tensor_1_square + tensor_2_square - 2 * multiply_sum

    # 最小下标计算
    min_index = torch.argmin(euclidean_distance)

    return tensor_2[min_index]


# m1, n1 = 10000, 1024
m1, n1 = 100000, 2048
tensor_1 = torch.rand((n1))
tensor_2 = torch.rand((m1,n1))

lp = LineProfiler()
lp_wrap = lp(get_elu_dis)
similar = lp_wrap(tensor_1, tensor_2)
print(lp.print_stats())
print(tensor_1)
print(tensor_2)
print(similar)
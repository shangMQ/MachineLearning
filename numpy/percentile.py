"""
python四分位距IQR计算
- 第一四分位数 (Q1)，又称“下四分位数”，等于该样本中所有数据由小到大排列后第25%的数据。
- 第二四分位数 (Q2)，又称“中位数”，等于该样本中所有数据由小到大排列后第50%数据。
- 第三四分位数 (Q3)，又称“上四分位数”，等于该样本中所有数据由小到大排列后第75%的数据。
- 第三四分位数与第一四分位数的差距又称四分位距（InterQuartile Range, IQR）
"""

import numpy as np
from scipy.stats import iqr

# case1
# 1. 使用np.percentile(数组，q几分位数（q1=25，q3=75），interpolation="linear"（对于取中位数，当中位数有两个数字时，选不同的参数来调整输出）)
a = np.array([1,2,3,4,5,6,7])
q1_a = np.percentile(a, 25)
q3_a = np.percentile(a, 75)
iqr_value = q3_a - q1_a
print("=====numpy.percentile=======")
print(f"iqr = {iqr_value}")
print(f"median = {np.percentile(a, 50)}")

# 2. 使用scipy.stats.iqr方法
iqr_value = iqr(a)
print("=====scipy.stats.iqr=======")
print(f"iqr = {iqr_value}")

# case2
# 1. 使用np.percentile(数组，q几分位数（q1=25，q3=75），interpolation="linear"（对于取中位数，当中位数有两个数字时，选不同的参数来调整输出）)
b = np.array([1.4, 2.6, 3.8, 4.7, 5.9, 6.2, 7.0])
q1_b = np.percentile(b, 25)
q3_b = np.percentile(b, 75)
iqr_value = q3_b - q1_b
print("=====numpy.percentile=======")
print(f"iqr = {iqr_value}")
print(f"median = {np.percentile(b, 50)}")

# 2. 使用scipy.stats.iqr方法
iqr_value = iqr(b)
print("=====scipy.stats.iqr=======")
print(f"iqr = {iqr_value}")

import numpy as np


array1 = np.array([1, 2])
array2 = np.array([[1, 1], [2, 3], [3,3]])
print("====== array1 =====")
print(array1)
print("====== array2 =====")
print(array2)

norm1 = np.power(array1, 2).sum(axis=0)
norm2 = np.power(array2, 2).sum(axis=1)
print("===== norm1 =====")
print(norm1)
print("===== norm2 =====")
print(norm2)
print("===== add =====")
print(norm1+norm2)

multiply_data = array1*array2
print("===== multiply data ======")
print(multiply_data)

print("==== sum =======")
sum_multiply = np.sum(multiply_data, axis=1)
print(sum_multiply)

print("===== euclidean distance =====")
euclidean = norm1 + norm2 - 2 * sum_multiply
print(euclidean)

print("==== argmin =======")
min_index = np.argmin(euclidean)
print(min_index)

print(f"similar data is {array2[min_index]}")




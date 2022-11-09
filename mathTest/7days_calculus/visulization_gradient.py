"""
可视化梯度——向量场(vector field)
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and y
x = np.linspace(-10, 10, 20)
xv, yv = np.meshgrid(x, x, indexing='ij')

# Compute the gradient of f(x,y)
fx = 2 * xv
fy = 2 * yv

# Convert the vector (fx,fy) into size and direction
size = np.sqrt(fx ** 2 + fy ** 2)
dir_x = fx / size
dir_y = fy / size

# Plot the surface
plt.figure(figsize=(6, 6))
# 绘制箭头
# 参数：X, Y, U, V, C
# X, Y 定义了箭头的位置, U, V 定义了箭头的方向, C 作为可选参数用来设置颜色
plt.quiver(xv, yv, dir_x, dir_y, size, cmap="viridis")
plt.colorbar()
plt.title("Vector Field")
plt.show()
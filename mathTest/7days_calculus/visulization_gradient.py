"""
可视化梯度——向量场(vector field)
"""

import numpy as np
import matplotlib.pyplot as plt

def vertor_field_plot():
    # vector field plot, take f(x,y) = x^2 + y^2 for example
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

def f(x, y):
    return x**2 + y**3

def find_best():
    # 考虑当前坐标，寻找梯度下降最快的方向
    # 0 to 360 degrees at 0.1-degree steps
    angles = np.arange(0, 360, 0.1)

    # coordinate to check
    # 假设当前在点(2,3)
    x, y = 2, 3

    # step size for differentiation
    step = 0.0001

    # To keep the size and direction of maximum rate of change
    # 遍历每个角度，找到梯度最大的角度
    maxdf, maxangle = -np.inf, 0
    for angle in angles:
        # convert degree to radian
        rad = angle * np.pi / 180
        # delta x and delta y for a fixed step size
        dx, dy = np.sin(rad) * step, np.cos(rad) * step
        # rate of change at a small step
        df = (f(x + dx, y + dy) - f(x, y)) / step
        # keep the maximum rate of change
        if df > maxdf:
            maxdf, maxangle = df, angle

    # Report the result
    dx, dy = np.sin(maxangle * np.pi / 180), np.cos(maxangle * np.pi / 180)
    gradx, grady = dx * maxdf, dy * maxdf
    print("===== calculate =====")
    print(f"Max rate of change at {maxangle} degrees")
    print(f"Gradient vector at ({x},{y}) is ({dx * maxdf},{dy * maxdf})")

    print("===== formula =====")
    print(f"Gradient vector at ({x},{y}) is ({2*x},{3*y**2})")

    print(f"tol = {abs(2*x - dx * maxdf) + abs(3*y**2 - dy * maxdf)}")

if __name__ == "__main__":
    # 向量场plot
    vertor_field_plot()
    # 计算当前节点的梯度值
    find_best()

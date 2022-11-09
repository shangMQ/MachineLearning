"""
双变量构成的函数可以用平面表示
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and y
x = np.linspace(-10, 10, 1000)
xv, yv = np.meshgrid(x, x, indexing='ij')

# Compute f(x,y) = x^2 + y^3
zv = xv ** 2 + yv ** 3

# Plot the surface
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xv, yv, zv, cmap="viridis")
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title("$f(x,y)=x^2+y^3$")
plt.show()

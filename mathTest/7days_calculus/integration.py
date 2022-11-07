"""
积分是微分的逆过程～

"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x

# Set up x from -10 to 10 with small steps
delta_x = 0.1
x = np.arange(-10, 10, delta_x)

# Find f(x) * delta_x
fx = f(x) * delta_x

# Compute the running sum
y = fx.cumsum()

# Plot
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("$f(x)=\int 2x = x^2$")
plt.show()
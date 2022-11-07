"""
antiderivative 反导数
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3 * x**2 - 4 * x

def antiderivative(f_x, delta_x, constant=0):
    fx = f_x * delta_x
    return fx.cumsum() + constant

# Set up x from -10 to 10 with small steps
delta_x = 0.1
x = np.arange(-2, 3, delta_x)

y = f(x)

antiderivative_value = antiderivative(y, delta_x)

plt.plot(x, antiderivative_value, "r-.")
plt.xlabel("x")
plt.title("$F(x)=\int 3x^2 - 4x dx$")
plt.show()

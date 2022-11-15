"""
梯度可以用来找到局部最优点
f(x)=log(x)的导数只能无限接近0
"""
import numpy as np
import matplotlib.pyplot as plt

# Define function f(x)
def f(x):
    return np.log(x)


# compute f(x) for x=-10 to x=10
x = np.linspace(-10, 10, 500)
y = f(x)

# f'(x) using the rate of change
delta_x = 0.0001
y_derivative = (f(x + delta_x) - f(x)) / delta_x


# Plot
fig = plt.figure(figsize=(12, 5))
plt.plot(x, y, 'k-', label="$f(x)=log(x)$")
plt.plot(x, y_derivative, c="r", alpha=0.5, label="$f'(x)= \\frac {1} {x}$")
# plt.vlines(optim_point, y.min(), y.max(), color='gray', linestyles='--')

plt.title("Derivative to find optim")
plt.xlabel("x")
plt.legend()

plt.show()
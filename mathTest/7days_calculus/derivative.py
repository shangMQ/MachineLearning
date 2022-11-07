import numpy as np
import matplotlib.pyplot as plt

# Define function f(x)
def f(x):
    return x ** 2

def plot(x):
    # plot initial f(x)
    plt.subplot(1,2,1)
    plt.plot(x, f(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$f(x)=x^2$")

    # f'(x) using the rate of change
    plt.subplot(1, 2, 2)
    delta_x = 0.0001
    y1 = (f(x + delta_x) - f(x)) / delta_x
    # f'(x) using the rule
    y2 = 2 * x
    # Plot f'(x) on right half of the figure
    plt.plot(x, y1, "r-", label="handy")
    plt.plot(x, y2, "b-.", label="$f(x)=2x$")
    plt.title("derivative calculation")
    plt.xlabel("x")
    plt.ylabel("")
    plt.legend()

    plt.suptitle("differentation")
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-10, 10, 500)
    plot(x)

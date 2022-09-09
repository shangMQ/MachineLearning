"""
蒙特卡洛仿真计算π
（深刻记得概率论老师讲过）
通过在 2x2 正方形内随机采样，我们可以使用以原点为中心的单位圆内包含的点的比例来估计圆的面积与正方形面积的比值。
鉴于我们知道真实比率为 π/4，我们可以将估计的比率乘以 4 来近似 π 的值。
采样的点越多，近似估计值越接近 π 的真实值。

圆的面积πr**2 / 正方形的面积(2r)**2 = π / 4，可以按照圆中元素的的个数m和总元素的个数n近似估计，得到π = 4*m/n

"""
import matplotlib.pyplot as plt
import numpy as np

def generate_data(sample_num, radis=1):
    a = 2 * radis
    b = 0 - radis
    out = np.random.rand(sample_num, 2) * a + b
    return out

def plot_data(initial_data, res_judge):
    fig = plt.figure(figsize=(5,5))
    circle_data = initial_data[res_judge==1]
    plt.scatter(initial_data[:,0], initial_data[:,1], color='green')
    plt.scatter(circle_data[:,0], circle_data[:,1], color='red')
    plt.title("monte_carlo")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

def check_in(data, res_judge, radis=1):
    def judge(item):
        if item[0]**2 + item[1]**2 <= radis**2:
            return 1
        else:
            return 0

    res_judge = np.apply_along_axis(judge, 1, data)
    return res_judge


if __name__ == "__main__":
    sample_num = 2000
    radis = 1
    data = generate_data(sample_num, radis)
    res_judge = check_in(data, radis)
    circle_num = sum(res_judge)

    plot_data(data, res_judge)
    pai = 4 * circle_num / sample_num
    print(f"pai = {pai}")

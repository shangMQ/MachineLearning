import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d #一定要引入这个包，不然会报错
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)


xy1, xy2 = np.meshgrid(x,y)  # 生成网格采样点
print(xy1.shape)

Z = [i**2 - j**2 for i, j in zip(xy1.ravel(), xy2.ravel())]
Z = np.array(Z).reshape(xy1.shape)

#2. 绘制3d saddle point
# plt.contourf(xy1, xy2, Z, cmap="Blues")
fig = plt.figure(figsize=(13, 7), facecolor='w')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xy1, xy2, Z, c='b', s=30, marker='o', depthshade=True)
ax.scatter(0, 0, 0, c='r', s=60)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("saddle plot")

plt.show()

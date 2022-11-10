"""
sympy提供了计算雅可比的方法
在雅可比矩阵中，每一行都有输出向量每个元素的偏微分，每一列都有关于输入向量每个元素的偏微分。
我们让 SymPy 定义符号x，y然后定义向量函数f。之后，可以通过调用该jacobian()函数找到雅可比行列式。


Sympy是一个符号计算的Python库。它的目标是成为一个全功能的计算机代数系统，同时保持代码简洁、易于理解和扩展。
它完全由Python写成，不依赖于外部库。SymPy支持符号计算、高精度计算、模式匹配、绘图、解方程、微积分、组合数学、离散 数学、几何学、
概率与统计、物理学等方面的功能。
"""

from sympy.abc import x, y
from sympy import Matrix, pprint, exp

# 对于f(x,y)=[2xy,x^2y]的雅可比矩阵的计算
f = Matrix([2*x*y, x**2*y])
variables = Matrix([x,y])
print("===Jacobian of f(x,y) = [2xy, x^2y]======")
pprint(f.jacobian(variables))

# 对于更为复杂一些的形如：f(x,y)=[1/(1+e^(-px+qy))]
g = Matrix([1/(1+exp(-x-y)), 1/(1+exp(-x-y)), 1/(1+exp(-x-y))])
variables = Matrix([x,y])
pprint(g.jacobian(variables))
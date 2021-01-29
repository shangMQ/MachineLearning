# -- coding = 'utf-8' --
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
累乘函数np.cumprod()测试
"""
import numpy as np

# 1. 默认情况：从左到右，从上到下依次累乘
prod1 = np.cumprod([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]])
print(prod1)

# 2. 指定axis=1 按行累乘
prod2 = np.cumprod([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]], axis=1)
print(prod2)

# 3. 指定axis=0 按列累乘
prod3 = np.cumprod([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]], axis=0)
print(prod3)




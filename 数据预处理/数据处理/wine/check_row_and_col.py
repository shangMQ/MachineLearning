# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
使用shell快速查看行和列数
"""
import os

filename = 'wine.data'

# 查看行数
row = os.popen(f"wc -l ./{filename}").read().split(' ')[-2]
print(f"row = {row}")

# 查看列数
col = len(os.popen(f'head -n 1 ./{filename}').read().split(','))
print(f"col = {col}")
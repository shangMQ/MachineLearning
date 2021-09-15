# -*- coding:utf-8 -*-
"""
dataGenerate

:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""

import random
import numpy as np
import pandas as pd

np.random.seed(100)

n_rows = 100000
n_features = 4
MAX_LENGTH = 25

# 1. ID
id_list = np.arange(1, n_rows+1).reshape(-1, 1)
id_data = pd.DataFrame(id_list)


# 2.中文数据生成
def Unicode(n):
    """
    生成若干中文
    """
    val = ""
    for i in range(n):
        val = val + chr(np.random.randint(0x4E00, 0x9FA5))
    return val


def generateUnicode(n_rows, n_features):
    unicodeLength = np.random.randint(1, MAX_LENGTH, size=(n_rows, n_features-1))
    print(unicodeLength)
    charlist= []
    for row in range(n_rows):
        row_list = []
        for n in range(n_features - 1):
            row_list.append(Unicode(unicodeLength[row][n]))
        charlist.append(row_list)
    return pd.DataFrame(charlist)


uni_data = generateUnicode(n_rows, n_features)
data = pd.concat([id_data, uni_data], axis=1)
data.to_csv("../data/initialData.csv", index=False)
# -*- coding:utf-8 -*-
"""
test

:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""

import pandas as pd
import numpy as np

from code.EncodeAndDecode import Encoder

data = pd.read_csv("../data/initialData.csv")

# 定义编码方式
CODING_STYLE = 'UTF-8'

# 定义编码字节个数
BYTES = 3

# 定义是否包含id字段
id_flag = True

# =====编码======
# 实例化编码对象
encoder = Encoder(CODING_STYLE, BYTES, id_flag)
encode_value = encoder.transform(data)
maxBytes_list = encoder.maxBytes_list

# 写入文件
encode_data = pd.DataFrame(encode_value)
encode_data.to_csv("../data/encode.csv", index=False)
# print(maxBytes_list)
maxBytes_data = pd.DataFrame(maxBytes_list)
maxBytes_data.to_csv("../data/encode_maxBytes.csv", index=False)

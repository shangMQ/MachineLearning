# -*- coding:utf-8 -*-
"""
decodeTest

:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""

import pandas as pd
import numpy as np

from code.EncodeAndDecode import Decoder

encode_data = pd.read_csv("../data/encode.csv")

# 定义编码方式
CODING_STYLE = 'UTF-8'

# 定义编码字节个数
BYTES = 3

# 定义是否包含id
id_flag = True

# 读取最大子节数
maxBytes_data = pd.read_csv("../data/encode_maxBytes.csv").values.ravel()

# =====解码=====
# 实例化解码对象
decoder = Decoder(CODING_STYLE, maxBytes_data, BYTES, id_flag)
decode_data = decoder.transform(encode_data)

# 写入文件
decode_data = pd.DataFrame(decode_data)
decode_data.to_csv("../data/decode.csv", index=False)
# -*- coding:utf-8 -*-
"""
EncodeAndDecode

:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""

import pandas as pd
import numpy as np


class Encoder():
    """
    中文字符串编码器：
    中文字符串-->int
    """
    def __init__(self, CODING_STYLE, BYTES, id_flag):
        """
        初始化
        :param CODING_STYLE: 编码类型
        :param BYTES: 编码类型对应字节数
        :param id_flag: 是否含id
        """
        self.CODING_STYLE = CODING_STYLE
        self.maxBytes_list = []
        self.BYTES = BYTES
        self.id_flag = id_flag

    def fit(self, data):
        """
        记录数据各个字段的最大字节数
        :param data:数据
        """
        if self.id_flag == True:
            # 如果包含id，编码字段就是除id列之后的其他字段
            data_columns = data.columns[1:]
        else:
            data_columns = data.columns[:]

        for col in data_columns:
            # 计算最大中文字符串长度
            max_length = data[col].str.len().max()
            # 计算字节
            max_bytes = max_length * self.BYTES
            # 记录最大子节
            self.maxBytes_list.append(max_bytes)

    def encodeFunc(self, s, max_bytes):
        """
        编码函数
        :param s: 字符串数据
        :param max_bytes: 最大字节数
        :return:编码后的字符数组
        """
        # 先对原始数据进行编码
        s_list = list(s.encode(encoding=self.CODING_STYLE, errors='strict'))
        # 为了后期便于处理，将较短的中文字符用0填充
        s_list += [0] * (max_bytes - len(s_list))
        return np.array(s_list)

    def transform(self, data):
        """
        编码
        :param data: pandas.DataFrame, 待转换的数据
        :return: pandas.DataFrame, 编码后的数据
        """
        self.fit(data)

        res_encoder = np.zeros((data.shape[0], 0), dtype=int)
        for i, col in enumerate(data.columns[1:]):
            # 对每列数据进行编码
            data[col] = data[col].apply(lambda x: self.encodeFunc(x, self.maxBytes_list[i]))
            # 合并数据
            res_encoder = np.hstack((res_encoder, np.vstack(data[col].values)))
            res_encoder = pd.DataFrame(res_encoder)

        if self.id_flag == True:
            # 如果含id, 在编码后，把列id加上
            id_data = data.iloc[:, 0]
            res_encoder = pd.concat([id_data, res_encoder], axis=1)

        return res_encoder


class Decoder():
    """
    中文字符串解码器：
    int-->中文字符串
    """
    def __init__(self, CODING_STYLE, maxBytes_list, BYTES, id_flag):
        """
        初始化
        :param CODING_STYLE: 编码类型
        :param maxBytes_list: 最大字节列表
        :param BYTES: 编码对应字节数
        :param id_flag: 是否含ID
        """
        self.CODING_STYLE = CODING_STYLE
        self.maxBytes_list = maxBytes_list
        self.BYTES = BYTES
        self.id_flag = id_flag

    def fit(self):
        """
        计算列宽索引（多少个数据合并为成中文字符串）
        :param data:数据
        :return: 列表分隔索引
        """
        col_index = np.cumsum(self.maxBytes_list)

        if self.id_flag == True:
            start = 1
            col_index += 1
        else:
            start = 0
        col_index = np.append(start, col_index)

        return col_index

    def decodeFunc(self, val):
        """
        对给定的数据进行解码操作
        :param val: 待解码的数据
        """
        decode_str = bytes(val).decode(encoding=self.CODING_STYLE)
        # 把\0换掉哈
        decode_str = decode_str.replace('\0', '')
        return decode_str

    def transform(self, data):
        """
        解码转换
        :param data: pandas.DataFrame, 待解码的数据
        :return: pandas.DataFrame, 解码后的数据
        """
        col_index = self.fit()
        res_decoder = pd.DataFrame()

        for i in range(1, len(col_index)):
            # 转换成列表，便于处理
            tmp = data.iloc[:, col_index[i - 1]:col_index[i]].values.tolist()
            res_decoder[str(i)] = tmp
            res_decoder[str(i)] = res_decoder[str(i)].apply(lambda x: self.decodeFunc(x))

        if self.id_flag == True:
            # 如果包含id，需要将id列加上
            id_data = data.iloc[:, 0]
            res_decoder = pd.concat([id_data, res_decoder], axis=1)

        return res_decoder


if __name__ == "__main__":
    dic = {'id': [0, 1, 2, 3], 'col1': ['嘻嘻', '哈', '哇哈哈哈', '大白菜'], 'col2': ['头盔', '装甲', '步枪', '狙击枪']}
    data = pd.DataFrame(dic)

    # 定义编码方式
    CODING_STYLE = 'UTF-8'

    # 定义编码字节个数
    BYTES = 3

    # 定义id
    id_flag = True

    # =====编码======
    # 实例化编码对象
    encoder = Encoder(CODING_STYLE, BYTES, id_flag)
    encode_value = encoder.transform(data)
    print(encode_value)
    maxBytes_list = encoder.maxBytes_list
    print(maxBytes_list)

    # =====解码=====
    # 实例化解码对象
    decoder = Decoder(CODING_STYLE, maxBytes_list, BYTES, id_flag)
    value = decoder.transform(encode_value)
    print(value)
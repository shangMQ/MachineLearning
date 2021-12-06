# -*- coding:utf-8 -*-
"""
manualGroupby 手动实现groupby的功能

:Author: kylin.smq@qq.com
:Last Modified by: kylin.smq@qq.com
"""

from operator import itemgetter
from collections import defaultdict
rows = [
    {'address': '5412 N CLARK', 'date': '07/01/2012'},
    {'address': '5148 N CLARK', 'date': '07/04/2012'},
    {'address': '5800 E 58TH', 'date': '07/02/2012'},
    {'address': '2122 N CLARK', 'date': '07/03/2012'},
    {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
    {'address': '1060 W ADDISON', 'date': '07/02/2012'},
    {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
    {'address': '1039 W GRANVILLE', 'date': '07/04/2012'}]


# groupby()函数扫描整个序列并且查找连续相同值（或者根据指定key函数返回值相同）的元素序列。
def groupby(items, key=None):
    # 构建一个多值字典
    d = defaultdict(list)
    for item in items:
        d[key(item)].append(item)

    # 在每次迭代的时候，它会返回一个值和一个迭代器对象
    for key, value in sorted(d.items()):
        yield (key, value)

# 1. 一个非常重要的准备步骤是要根据指定的字段将数据排序
rows = sorted(rows, key = itemgetter('date'))
groupby(rows, key=itemgetter('date'))

# 2. 查看groupby效果
for date, items in groupby(rows, key=itemgetter('date')):
    print(date)
    for item in items:
        print(item)

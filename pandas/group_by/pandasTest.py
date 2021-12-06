# -*- coding:utf-8 -*-
"""
pandasTest GroupBy的使用
当我们根据某个字段进行group机制分组的时候，最后能够生成多少个子DataFrame，取决于我们的字段中有多少个不同的元素；
当我们分组之后，便可以进行后续的各种聚合操作，比如sum、mean、min等。

:Author: kylin.smq@qq.com
:Last Modified by: kylin.smq@qq.com
"""

import pandas as pd
import numpy as np

# 构造数据
num = 10
ids = range(num)
np.random.seed(0)
scores = np.random.randint(0, 5, num)
employees = pd.DataFrame({'id': ids, 'scores': scores, 'salary': np.random.randint(1000, 9999, num)})
print("Tab. Employee information")
print(employees)

# Test1: 按照scores进行分组
print("=======Group By Scores=======")
# 1. 得到的是一个DataFrameGroupBy对象
scores_gb = employees.groupby(by='scores')
print(scores_gb)
# 1.1 可以用list展开查看该对象的内容
# 根据分组字段划分得到各个元组，里面存放着相关数据的信息
print(list(scores_gb)[0])
# 1.2 可以遍历GroupBy对象
print("******Traverse Scores_GroupBy Object******")
for score, group in scores_gb:
    print(score)
    print(group)
# 1.3 对DataFrameGroupBy对象使用get_group()方法，能够让我们得到分组元素中的指定组的数据
print("******Using get_group method to select the items which score equals zero******")
print(scores_gb.get_group(0))

# 2. 然后选择需要研究的列， 返回的是一个SeriesGroupBy对象，不易直观查看
# agg是根据分组字段和聚合函数生成新的数据帧
print("=======Select columns for aggregating=======")
# 2.1 例如选择查看按照score进行分组的相关员工的薪水情况
print(scores_gb['salary'])
print(pd.DataFrame(scores_gb['salary']))
# 2.2 对SeriesGroupby进行操作, 比如.mean(), 相当于对每个组的Series求均值
print("******Mean******")
print(scores_gb['salary'].mean())
# 分组之后对同一个列名使用不同的函数，函数使用列表形式：下面是对score分别求和、最大值、最小值、均值、个数（size）
print("*****Check aggregating function*******")
agg_res1 = employees.groupby("scores")["salary"].agg(["sum","max","min","mean","size"]).reset_index()
print(agg_res1)

# 3. 使用transform将聚合结果添加至原数据中
# transform是在原数据基础上加新的列，需要指定相关的聚合函数
print("=======Using transform to append the aggregating result=======")
employees['salary_mean'] = employees.groupby('scores')['salary'].transform("mean")
print(employees)
# 不使用transform的实现
avg_salary = employees.groupby('scores')['salary'].mean().to_dict()
print(avg_salary)
employees['salary_avg'] = employees['scores'].map(avg_salary)
print(employees)
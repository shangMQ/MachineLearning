# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
模糊匹配库相关练习
"""
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# 1. 简单匹配
test1 = fuzz.ratio("Kylin is ysl", "kylin is ysl")
print("'Kylin is ysl' VS 'kylin is ysl' = ", test1)

# 2. 非完全匹配
test2 = fuzz.partial_ratio("Kylin is ysl", "Kylin is ysl!")
print("'Kylin is ysl' VS 'Kylin is ysl!' = ", test2)

# 3. 去重子集匹配
test3 = fuzz.token_sort_ratio("Kylin is ysl", "Kylin Kylin is ysl")
print("'Kylin is ysl' VS 'Kylin Kylin is ysl' = ", test3)

# process用来返回模糊匹配的字符串和相似度
choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
print(process.extract("new york jets", choices, limit=2))
# 传入附加参数到extractOne方法来设置使用特定的匹配模式并返回和目标匹配的字符串相似度最高的字符串。
print(process.extractOne("cowboys", choices))

# coding=utf-8
# 通过横截面找因子

import pandas as pd
from research.future.alpha_auto.alpha_auto.function.technique import  MultiIndexMethod
from research.future.alpha_auto.alpha_auto.alpha.alpha_rank_auto_produce import AlphaRankAutoProduce

file_path = "C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\alpha_auto\\example\\day.csv"
df1 = pd.read_csv(file_path, index_col=[0, 2], skipinitialspace=True)

df = MultiIndexMethod.get_filter_symbols(df1, 365*2)
a = AlphaRankAutoProduce(df, canshu_nums=1)
a.run()


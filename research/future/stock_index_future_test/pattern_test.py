import os
import pandas as pd
from deap import gp
import datetime
import numpy as np

import model.constants.path as ConstPath
import model.constants.genetic as ConstGenetic
from model.genetic import GP
from model.genetic import config

import sprt as sprt
import numpy as np

# Null value
h0 = 0
# Alternative value
h1 = 0.2
# Type I error rate = 0.05
alpha = 0.05
# Type II error rate = 0.2
beta = 0.05
# Values
values = np.random.normal(0, 1, 100)
test = sprt.SPRTBinomial(h0=h0, h1=h1, alpha=alpha, beta=beta, values=values)

if __name__ == '__main__':
    file_path = 'C:\\Users\\jason.huang\\research\\data_mining\\GP\\mining_result\\'
    result_path = [x for x in os.listdir(file_path) if 'pattern_recognition_every_other_row_2022_1017_17' in x]
    data_all = pd.DataFrame()
    for name1 in result_path:
        file_location = file_path + name1 + '\\name_dict.csv'
        data = pd.read_csv(file_location, index_col=0)
        data['path'] = file_path + name1
        data_all = pd.concat([data_all, data])
    data_all = data_all.reset_index(drop=True)

    data_all = data_all[data_all['IR_all_data'] > 0.5]
    data_all = data_all[data_all['Ret_all_data'] > 0]

    data_all = data_all.reset_index(drop=True)
    data_all = data_all.drop_duplicates('Formula')

    data_all = data_all.sort_values('IR_all_data', ascending=False).reset_index(drop=True)

    config_dict_pattern = config.config_dict_pattern
    config_dict_pattern['end_date'] = '2022-10-30'
    gp_obj = GP.GeneticProgramming(config_dict=config_dict_pattern)
    data_obj_func_list = []
    non_data_obj_func_list = []
    for func in gp_obj.config[ConstGenetic.config_dict_key_gp_operator]:
        # import funcition from the function py module
        exec(f"from model.genetic.function.gp_operator import {func}")
        if 'data_obj' in eval(func).__annotations__.keys():
            data_obj_func_list.append(func)
        else:
            non_data_obj_func_list.append(func)
    trade_num = min(len(gp_obj.config[ConstGenetic.config_dict_key_tickers]), 10)

    for i in range(len(data_all)):
        func_str = data_all.loc[i, "Formula"]
        func_str = func_str.replace(")", ",data_obj=data_obj_we_need_to_change)")
        func_str = func_str.replace("(,", "(")
        for str1 in non_data_obj_func_list:
            func_str = func_str.replace(str1 + "(data_obj=data_obj_we_need_to_change)", str1 + "()")

        ###########################################################################
        # all data
        I = gp_obj.all_data.I
        res = eval(func_str.replace("data_obj_we_need_to_change", "gp_obj.all_data"))
        pct = gp_obj.all_data.ret
        nv = res.shift(2).mul(pct).cumsum()
        nv.index = gp_obj.all_data.time_index
        nv.index_name = func_str

        volume = getattr(gp_obj.all_data, "volume")
        res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
        volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)

        ret_cond = pct.shift(-2).where(volumerank <= trade_num)
        ret_mean = ret_cond.sum(axis=1) / trade_num
        ret_mean.index = gp_obj.all_data.time_index
        nv['strategy'] = ret_mean.cumsum().values
        nv.to_csv(
            "C:\\Users\\jason.huang\\research\\data_mining\\GP\\mining_result\\selected_long\\Factor_" + str(
                i) + '.csv')

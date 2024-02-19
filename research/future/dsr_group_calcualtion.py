import sys

sys.path.append('..')
sys.path.append('/home/pinpoint/Documents/code/')
from model.genetic.function.result_save_function import GeneticProgrammingResult
from model.genetic.function.result_analysis_tool import NetValueAnalysis
from model.genetic.function.result_analysis_tool import single_factor_pickle_to_df_parallel,single_factor_pickle_weight_to_dict_parallel,pickle_load_parallel
from model.genetic.function.sharpe_ratio_test import skew_to_alpha, moments, estimated_sharpe_ratio, ann_estimated_sharpe_ratio, probabilistic_sharpe_ratio, num_independent_trials, expected_maximum_sr, deflated_sharpe_ratio

from scipy.stats import skewnorm, norm
import pickle
import sys
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import os
from collections import ChainMap

plt.rcParams['figure.figsize'] = [16,9]


out_of_sample_time='2022-01-01'

# root_file_path='/home/pinpoint/Documents/result/gp_result/commodity_future_selected_daily/'
# tickers_group_list=['selected_1','selected_2','selected_3','selected_4','selected_5']


root_file_path='/home/pinpoint/Documents/result/gp_result/financial_future_daily/'
tickers_group_list=['stock_index','interest_rate']


# root_file_path='/home/pinpoint/Documents/result/gp_result/stock_index_test/'
# tickers_group_list=['stock_index']
schema_list=['ts_factor','pr_long','pr_short']


print(f"all data number {len(os.listdir(root_file_path))}")
for tickers_group_name in tickers_group_list:
    for schema_name in schema_list:
        print(f"{tickers_group_name} {schema_name} number: {len([x for x in os.listdir(root_file_path) if tickers_group_name in x and schema_name in x])}")

tickers_group_name="stock_index"
schema_name="ts_factor"

selected_file_path_list=[x for x in os.listdir(root_file_path) if tickers_group_name in x and schema_name in x]
indicator_all=pd.DataFrame()


file_location_list=[root_file_path+x+'/total.pkl' for x in selected_file_path_list]
print(len(file_location_list))

import random
sample_size=len(file_location_list)
# sample_size=500
file_location_list=random.sample(file_location_list,sample_size)
df_list=pickle_load_parallel(file_location_list=file_location_list)
print(f"LOAD SAMPLE {sample_size} SUMMARY")
indicator_all = pd.concat(df_list, ignore_index=True)
print(f"indicator_all shape: {indicator_all.shape}")

indicator_all=indicator_all.drop_duplicates(['Formula'])
indicator_all=indicator_all.drop_duplicates(['Ret_total','IR_total','Calmar_total'])
selected_indicator=indicator_all.copy()
selected_indicator=indicator_all[indicator_all['IR_train']>1]
selected_indicator=selected_indicator[selected_indicator['Calmar_test']>1]


print(f"selected_indicator shape: {selected_indicator.shape}")

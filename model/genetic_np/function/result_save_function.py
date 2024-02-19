import time

from deap import gp
import datetime
import os
import pandas as pd
import numpy as np
import pickle

import model.constants.path as ConstPath
import model.constants.genetic as ConstGenetic
from model.genetic.GP import GeneticProgrammingData
import model.constants.futures as ConstFut


# from model.genetic import GP


#
# # save factor result
# def save_ts_factor_ic(gp_obj, result):
#     name_dict = pd.DataFrame()
#     t = gp_obj.config['gp_schema'] + '_' + gp_obj.config['split_method']['method'] + '_' + datetime.datetime.strftime(
#         datetime.datetime.now(), '%Y_%m%d_%H%M%S_%f')
#     save_path = ConstPath.output_factor_path + t + '\\'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     for i in range(len(result)):
#         tree = gp.PrimitiveTree(result[i])
#         print(str(tree))
#         function = gp.compile(tree, gp_obj.pset)
#         res = eval(
#             f"function({','.join([i + '=' + 'gp_obj.all_data.' + i for i in gp_obj.config[ConstGenetic.config_dict_key_feature]])})")
#         res.index.name = str(tree)
#         res.index = gp_obj.all_data.time_index
#         res.to_csv(save_path + 'Factor_' + str(i) + '_all_data.csv')
#
#         res.index = range(len(res))
#         name_dict.loc[i, 'Factor'] = 'Factor_' + str(i)
#         name_dict.loc[i, 'Formula'] = str(tree)
#         ic_result = res.corrwith(gp_obj.all_data.ret)
#         name_dict.loc[i, 'IC_mean_abs_all_data'] = ic_result.abs().mean()
#         name_dict.loc[i, 'IC_median_abs_all_data'] = ic_result.abs().median()
#         try:
#             name_dict.loc[i, 'IC_mean_max_5_all_data'] = ic_result.sort_values()[-5:].mean()
#         except:
#             name_dict.loc[i, 'IC_mean_max_5_all_data'] = np.nan
#         try:
#             name_dict.loc[i, 'IC_mean_min_5_all_data'] = ic_result.sort_values()[:5].mean()
#         except:
#             name_dict.loc[i, 'IC_mean_min_5_all_data'] = np.nan
#
#         res.index = gp_obj.all_data.time_index
#
#         split_param_list = [str(k) + '=' + str(v) for k, v in
#                             gp_obj.config[ConstGenetic.config_dict_key_train_test_split_method].items()]
#         param_str = ','.join(split_param_list[1:])
#         method = split_param_list[0].split('=')[1]
#         exec(f"from model.genetic.function.data_split_function import {method}")
#         if len(param_str) == 0:
#             res_train, res_test = eval(f"{method}(data=res)")
#         else:
#             res_train, res_test = eval(f"{method}(data=res,{param_str})")
#
#         # train data result record
#         res = res_train.copy().reset_index(drop=True)
#         ic_result = res.corrwith(gp_obj.train_data.ret)
#         name_dict.loc[i, 'IC_mean_abs_train_data'] = ic_result.abs().mean()
#         name_dict.loc[i, 'IC_median_abs_train_data'] = ic_result.abs().median()
#         try:
#             name_dict.loc[i, 'IC_mean_max_5_train_data'] = ic_result.sort_values()[-5:].mean()
#         except:
#             name_dict.loc[i, 'IC_mean_max_5_train_data'] = np.nan
#         try:
#             name_dict.loc[i, 'IC_mean_min_5_train_data'] = ic_result.sort_values()[:5].mean()
#         except:
#             name_dict.loc[i, 'IC_mean_min_5_train_data'] = np.nan
#
#         # test data result record
#         res = res_test.copy().reset_index(drop=True)
#         ic_result = res.corrwith(gp_obj.test_data.ret)
#         name_dict.loc[i, 'IC_mean_abs_test_data'] = ic_result.abs().mean()
#         name_dict.loc[i, 'IC_median_abs_test_data'] = ic_result.abs().median()
#         try:
#             name_dict.loc[i, 'IC_mean_max_5_test_data'] = ic_result.sort_values()[-5:].mean()
#         except:
#             name_dict.loc[i, 'IC_mean_max_5_test_data'] = np.nan
#         try:
#             name_dict.loc[i, 'IC_mean_min_5_test_data'] = ic_result.sort_values()[:5].mean()
#         except:
#             name_dict.loc[i, 'IC_mean_min_5_test_data'] = np.nan
#         name_dict.to_csv(save_path + 'name_dict.csv')
#
#
# # save pattern strategy result
# def save_ts_pattern_recognition_nv(gp_obj, result):
#     name_dict = pd.DataFrame()
#     t = gp_obj.config['gp_schema'] + '_' + gp_obj.config['split_method']['method'] + '_' + datetime.datetime.strftime(
#         datetime.datetime.now(), '%Y_%m%d_%H%M%S_%f')
#     save_path = ConstPath.output_factor_path + t + '\\'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     trade_num = min(len(gp_obj.config[ConstGenetic.config_dict_key_tickers]), 10)
#
#     config_dict = gp_obj.config
#     config_dict[ConstGenetic.config_dict_key_end_date] = '2024-01-01'
#     out_of_sample = GeneticProgrammingData(config_dict=config_dict)
#
#     for i in range(len(result)):
#         tree = gp.PrimitiveTree(result[i])
#         print(str(tree))
#
#         name_dict.loc[i, 'Factor'] = 'Factor_' + str(i)
#         name_dict.loc[i, 'Formula'] = str(tree)
#
#         data_obj_func_list = []
#         non_data_obj_func_list = []
#         for func in gp_obj.config[ConstGenetic.config_dict_key_gp_operator]:
#             # import funcition from the function py module
#             exec(f"from model.genetic.function.gp_operator import {func}")
#             if 'data_obj' in eval(func).__annotations__.keys():
#                 data_obj_func_list.append(func)
#             else:
#                 non_data_obj_func_list.append(func)
#         func_str = str(tree).replace(")", ",data_obj=data_obj_we_need_to_change)")
#         func_str = func_str.replace("(,", "(")
#         for str1 in non_data_obj_func_list:
#             func_str = func_str.replace(str1 + "(data_obj=data_obj_we_need_to_change)", str1 + "()")
#
#         ###########################################################################
#         # all data
#         I = gp_obj.all_data.I
#         res = eval(func_str.replace("data_obj_we_need_to_change", "gp_obj.all_data"))
#         pct = gp_obj.all_data.ret.reset_index(drop=True)
#         nv = res.shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret = res.shift(2).mul(pct)
#         ret_mean = ret.sum(axis=1) / len(ret.columns)
#         ret_mean.index = gp_obj.all_data.time_index
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\all_nv\\'):
#             os.makedirs(save_path + '\\all_nv\\')
#
#         nv.to_csv(save_path + '\\all_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_all_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_all_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_all_data'] = np.nan
#
#         try:
#             name_dict.loc[i, 'Calmar_all_data'] = ret_mean.sum() / (
#                     np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min()) + 0.001)
#         except:
#             name_dict.loc[i, 'Calmar_all_data'] = np.nan
#         name_dict.loc[i, 'Turnover_train_data'] = (res.diff().abs().sum() / len(res)).mean()
#
#         res.index = gp_obj.all_data.time_index
#
#         split_param_list = [str(k) + '=' + str(v) for k, v in
#                             gp_obj.config[ConstGenetic.config_dict_key_train_test_split_method].items()]
#         param_str = ','.join(split_param_list[1:])
#         method = split_param_list[0].split('=')[1]
#         exec(f"from model.genetic.function.data_split_function import {method}")
#         if len(param_str) == 0:
#             res_train, res_test = eval(f"{method}(data=res)")
#         else:
#             res_train, res_test = eval(f"{method}(data=res,{param_str})")
#
#         res.index.name = str(tree)
#         res.to_csv(save_path + 'Factor_' + str(i) + '.csv')
#
#         pct = gp_obj.all_data.ret
#         pct.index = gp_obj.all_data.time_index
#
#         # train data
#         res = res_train.copy()
#         nv = res.shift(2).mul(pct).cumsum()
#         nv.index_name = str(tree)
#         ret_mean = res.shift(2).mul(pct).sum(axis=1).div(len(pct.columns))
#
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\train_nv\\'):
#             os.makedirs(save_path + '\\train_nv\\')
#
#         nv.to_csv(save_path + '\\train_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_train_data'] = ret_mean.dropna().mean() * 252
#         try:
#             name_dict.loc[i, 'IR_train_data'] = ret_mean.dropna().mean() / ret_mean.dropna().std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_train_data'] = np.nan
#         try:
#             name_dict.loc[i, 'Calmar_all_data'] = ret_mean.sum() / (
#                     np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min()) + 0.001)
#         except:
#             name_dict.loc[i, 'Calmar_all_data'] = np.nan
#         name_dict.loc[i, 'Turnover_train_data'] = (res.diff().abs().sum() / len(res)).mean()
#
#         # test data
#
#         res = res_test.copy()
#         nv = res.shift(2).mul(pct).cumsum()
#         nv.index_name = str(tree)
#         ret_mean = res.shift(2).mul(pct).sum(axis=1).div(len(pct.columns))
#
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\test_nv\\'):
#             os.makedirs(save_path + '\\test_nv\\')
#
#         nv.to_csv(save_path + '\\test_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_test_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_test_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_test_data'] = np.nan
#         try:
#             name_dict.loc[i, 'Calmar_all_data'] = ret_mean.sum() / (
#                     np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min()) + 0.001)
#         except:
#             name_dict.loc[i, 'Calmar_all_data'] = np.nan
#         name_dict.loc[i, 'Turnover_train_data'] = (res.diff().abs().sum() / len(res)).mean()
#
#         name_dict.to_csv(save_path + 'name_dict.csv')
#
#
# #
# def save_ts_factor_strategy_nv(gp_obj, result):
#     name_dict = pd.DataFrame()
#     t = gp_obj.config['gp_schema'] + '_' + gp_obj.config['split_method']['method'] + '_' + datetime.datetime.strftime(
#         datetime.datetime.now(), '%Y_%m%d_%H%M%S_%f')
#     save_path = ConstPath.output_factor_path + t + '\\'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     config_dict = gp_obj.config
#     config_dict[ConstGenetic.config_dict_key_end_date] = '2024-01-01'
#     out_of_sample = GeneticProgrammingData(config_dict=config_dict)
#
#     for i in range(len(result)):
#         tree = gp.PrimitiveTree(result[i])
#         print(str(tree))
#         function = gp.compile(tree, gp_obj.pset)
#         res = eval(
#             f"function({','.join([i + '=' + 'gp_obj.all_data.' + i for i in gp_obj.config[ConstGenetic.config_dict_key_feature]])})")
#
#         name_dict.loc[i, 'Factor'] = 'Factor_' + str(i)
#         name_dict.loc[i, 'Formula'] = str(tree)
#
#         pct = gp_obj.all_data.ret
#         nv = res.applymap(np.sign).shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.applymap(np.sign).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\all_nv\\'):
#             os.makedirs(save_path + '\\all_nv\\')
#
#         nv.to_csv(save_path + '\\all_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_all_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_all_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_all_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_all_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_all_data'] = np.nan
#
#         res.index = gp_obj.all_data.time_index
#
#         split_param_list = [str(k) + '=' + str(v) for k, v in
#                             gp_obj.config[ConstGenetic.config_dict_key_train_test_split_method].items()]
#         param_str = ','.join(split_param_list[1:])
#         method = split_param_list[0].split('=')[1]
#         exec(f"from model.genetic.function.data_split_function import {method}")
#         if len(param_str) == 0:
#             res_train, res_test = eval(f"{method}(data=res)")
#         else:
#             res_train, res_test = eval(f"{method}(data=res,{param_str})")
#
#         res.index.name = str(tree)
#         res.to_csv(save_path + 'Factor_' + str(i) + '.csv')
#
#         # train data
#         res = res_train.copy().reset_index(drop=True)
#         pct = gp_obj.train_data.ret
#         nv = res.applymap(np.sign).shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.train_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.applymap(np.sign).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\train_nv\\'):
#             os.makedirs(save_path + '\\train_nv\\')
#
#         nv.to_csv(save_path + '\\train_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_train_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_train_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_train_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_train_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_train_data'] = np.nan
#
#         # test data
#         res = res_test.copy().reset_index(drop=True)
#         pct = gp_obj.test_data.ret
#         nv = res.applymap(np.sign).shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.test_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.applymap(np.sign).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\test_nv\\'):
#             os.makedirs(save_path + '\\test_nv\\')
#
#         nv.to_csv(save_path + '\\test_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_test_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_test_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_test_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_test_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_test_data'] = np.nan
#
#         # out of sample total
#         res = eval(
#             f"function({','.join([i + '=' + 'out_of_sample.all_data.' + i for i in gp_obj.config[ConstGenetic.config_dict_key_feature]])})")
#         pct = out_of_sample.all_data.ret
#         pct = pct.reset_index(drop=True)
#         nv = res.applymap(np.sign).shift(2).mul(pct).cumsum()
#         nv.index = out_of_sample.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.applymap(np.sign).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\total_nv\\'):
#             os.makedirs(save_path + '\\total_nv\\')
#
#         nv.to_csv(save_path + '\\total_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_total_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_total_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_total_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_total_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_total_data'] = np.nan
#
#         # out of sample data
#         res.index = out_of_sample.all_data.time_index
#         res = res.drop(index=gp_obj.all_data.time_index)
#         pct = out_of_sample.all_data.ret
#         pct.index = out_of_sample.all_data.time_index
#         pct = pct.drop(index=gp_obj.all_data.time_index)
#
#         nv = res.applymap(np.sign).shift(2).mul(pct).cumsum()
#         # nv.index = out_of_sample.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.applymap(np.sign).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\out_of_sample_nv\\'):
#             os.makedirs(save_path + '\\out_of_sample_nv\\')
#
#         nv.to_csv(save_path + '\\out_of_sample_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_out_of_sample_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_out_of_sample_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_out_of_sample_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_out_of_sample_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_out_of_sample_data'] = np.nan
#
#         name_dict.to_csv(save_path + 'name_dict.csv')
#
#
# #
# def save_ts_factor_weight_function_phi_nv(gp_obj, result):
#     name_dict = pd.DataFrame()
#     t = gp_obj.config['gp_schema'] + '_' + gp_obj.config['split_method']['method'] + '_' + datetime.datetime.strftime(
#         datetime.datetime.now(), '%Y_%m%d_%H%M%S_%f')
#     save_path = ConstPath.output_factor_path + t + '\\'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     config_dict = gp_obj.config
#     config_dict[ConstGenetic.config_dict_key_end_date] = '2024-01-01'
#     out_of_sample = GeneticProgrammingData(config_dict=config_dict)
#
#     for i in range(len(result)):
#         tree = gp.PrimitiveTree(result[i])
#         print(str(tree))
#         function = gp.compile(tree, gp_obj.pset)
#         res = eval(
#             f"function({','.join([i + '=' + 'gp_obj.all_data.' + i for i in gp_obj.config[ConstGenetic.config_dict_key_feature]])})")
#
#         name_dict.loc[i, 'Factor'] = 'Factor_' + str(i)
#         name_dict.loc[i, 'Formula'] = str(tree)
#
#         pct = gp_obj.all_data.ret
#         nv = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\all_nv\\'):
#             os.makedirs(save_path + '\\all_nv\\')
#
#         nv.to_csv(save_path + '\\all_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_all_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_all_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_all_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_all_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_all_data'] = np.nan
#
#         res.index = gp_obj.all_data.time_index
#
#         split_param_list = [str(k) + '=' + str(v) for k, v in
#                             gp_obj.config[ConstGenetic.config_dict_key_train_test_split_method].items()]
#         param_str = ','.join(split_param_list[1:])
#         method = split_param_list[0].split('=')[1]
#         exec(f"from model.genetic.function.data_split_function import {method}")
#         if len(param_str) == 0:
#             res_train, res_test = eval(f"{method}(data=res)")
#         else:
#             res_train, res_test = eval(f"{method}(data=res,{param_str})")
#
#         res.index.name = str(tree)
#         res.to_csv(save_path + 'Factor_' + str(i) + '.csv')
#
#         # train data
#         res = res_train.copy().reset_index(drop=True)
#         pct = gp_obj.train_data.ret
#         nv = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.train_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\train_nv\\'):
#             os.makedirs(save_path + '\\train_nv\\')
#
#         nv.to_csv(save_path + '\\train_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_train_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_train_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_train_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_train_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_train_data'] = np.nan
#
#         # test data
#         res = res_test.copy().reset_index(drop=True)
#         pct = gp_obj.test_data.ret
#         nv = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).cumsum()
#         nv.index = gp_obj.test_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\test_nv\\'):
#             os.makedirs(save_path + '\\test_nv\\')
#
#         nv.to_csv(save_path + '\\test_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_test_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_test_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_test_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_test_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_test_data'] = np.nan
#
#         # out of sample total
#         res = eval(
#             f"function({','.join([i + '=' + 'out_of_sample.all_data.' + i for i in gp_obj.config[ConstGenetic.config_dict_key_feature]])})")
#         pct = out_of_sample.all_data.ret
#         pct = pct.reset_index(drop=True)
#         nv = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).cumsum()
#         nv.index = out_of_sample.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\total_nv\\'):
#             os.makedirs(save_path + '\\total_nv\\')
#
#         nv.to_csv(save_path + '\\total_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_total_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_total_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_total_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_total_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_total_data'] = np.nan
#
#         # out of sample data
#         res.index = out_of_sample.all_data.time_index
#         res = res.drop(index=gp_obj.all_data.time_index)
#         pct = out_of_sample.all_data.ret
#         pct.index = out_of_sample.all_data.time_index
#         pct = pct.drop(index=gp_obj.all_data.time_index)
#
#         nv = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).cumsum()
#         # nv.index = out_of_sample.all_data.time_index
#         nv.index_name = str(tree)
#
#         ret_mean = res.fillna(0).applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2).mul(pct).mean(axis=1)
#         nv['strategy'] = ret_mean.cumsum().values
#
#         if not os.path.exists(save_path + '\\out_of_sample_nv\\'):
#             os.makedirs(save_path + '\\out_of_sample_nv\\')
#
#         nv.to_csv(save_path + '\\out_of_sample_nv\\Factor_' + str(i) + '.csv')
#
#         name_dict.loc[i, 'Ret_out_of_sample_data'] = ret_mean.mean() * 252
#         try:
#             name_dict.loc[i, 'IR_out_of_sample_data'] = ret_mean.mean() / ret_mean.std() * 15.8
#         except:
#             name_dict.loc[i, 'IR_out_of_sample_data'] = np.nan
#         try:
#             mdd = np.abs((ret_mean.cumsum() - ret_mean.cumsum().cummax()).min())
#             name_dict.loc[i, 'Calmar_out_of_sample_data'] = ret_mean.sum() / (mdd + 0.0001)
#         except:
#             name_dict.loc[i, 'Calmar_out_of_sample_data'] = np.nan
#
#         name_dict.to_csv(save_path + 'name_dict.csv')


class NetValueAnalysis:
    def __init__(self, nv):
        self.nv = nv
        self.stats = self._calculate_stats()

    def _calculate_stats(self):
        try:
            nv_all = self.nv.copy()
            nv_all = pd.DataFrame(nv_all)
            nv_all['Date'] = [datetime.datetime.strftime(pd.to_datetime(nv_all.index[x]), "%Y%m%d") for x in
                              range(len(nv_all))]
            max_num = nv_all.groupby('Date')['Date'].count().max()
        except:
            max_num = 1

        def __calculate_annual_return(ret):
            return ret.mean() * 252 * np.sqrt(max_num)

        def __calculate_annual_ir(ret):
            return ret.mean() / (ret.std() + 0.0001) * 15.8 * np.sqrt(max_num)

        def __calculate_calmar(ret):
            mdd = (ret.cumsum() - ret.cumsum().cummax()).min()
            return ret.mean() / (np.abs(mdd) + 0.001) * 252 * np.sqrt(max_num)

        stats = pd.DataFrame()
        stats.loc['Total', 'Ret'] = __calculate_annual_return(self.nv)
        stats.loc['Total', 'IR'] = __calculate_annual_ir(self.nv)
        stats.loc['Total', 'Calmar'] = __calculate_calmar(self.nv)

        return stats

    def plot_strategy_nv(self):
        import matplotlib.pyplot as plt
        stats_show = self.stats.copy()
        for i in range(len(stats_show)):
            stats_show.iloc[i, 0] = str(round(stats_show.iloc[i, 0] * 100, 2)) + '%'
            stats_show.iloc[i, 1] = str(round(stats_show.iloc[i, 1], 2))
            stats_show.iloc[i, 2] = str(round(stats_show.iloc[i, 2], 2))
        fig, axs = plt.subplots(3, 1, figsize=(20, 30))

        axs[0].axis('off')
        the_table = axs[0].table(cellText=stats_show.values, colLabels=stats_show.columns, rowLabels=stats_show.index,
                                 cellLoc='center', loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(20)
        the_table.scale(1, 4)

        axs[1].plot(self.nv.fillna(method='ffill').fillna(0), label='strategy')
        axs[1].set_title("strategy nv and undetlying nv")

        axs[2].plot(self.nv - self.nv.cummax(), lw=3, label='drawdown', alpha=0.5)
        axs[2].fill_between(x=self.nv.index, y1=np.array([0] * len(self.nv)),
                            y2=self.nv - self.nv.cummax(), alpha=0.7)
        axs[2].legend()
        axs[2] = axs[2].twinx()
        axs[2].plot(
            self.nv.diff().rolling(60).mean() / self.nv.diff().rolling(60).std(),
            label='rolling sharpe', color='red', alpha=0.5)
        axs[2].legend()
        axs[2].set_title('Drawdown and Rolling Sharpe')


class GeneticProgrammingResult:
    def __init__(self, config_dict, formula):
        self.gp_obj = GeneticProgrammingData(config_dict)
        self.config = config_dict
        self.formula = str(formula)
        formula_gp_form = formula

        self.factor_value = self._calculate_factor_value(gp_obj=self.gp_obj, formula=formula_gp_form)
        self.factor_weighting_value = self._calculate_factor_weighting_value(gp_obj=self.gp_obj,
                                                                             formula=formula_gp_form)
        self.sub_nv = self._calculate_ret(gp_obj=self.gp_obj, formula=formula_gp_form).cumsum()
        self.strategy_nv = self._calculate_strategy_ret(gp_obj=self.gp_obj, formula=formula_gp_form).cumsum()
        self.strategy_benchmark_nv = self._calculate_strategy_benchmark_ret(gp_obj=self.gp_obj,
                                                                            formula=formula_gp_form).cumsum()
        self.stats = self._calculate_stats(gp_obj=self.gp_obj, formula=formula_gp_form)

        self.name_dict = self.__name_dict()

    def _calculate_factor_value(self, gp_obj=None, formula=None, data_type='total_data'):
        if gp_obj is None:
            gp_obj = self.gp_obj

        if ConstGenetic.config_dict_value_schema_pattern_recognition in gp_obj.config[
            ConstGenetic.config_dict_key_schema]:
            data_obj_func_list = []
            non_data_obj_func_list = []
            for func in gp_obj.config[ConstGenetic.config_dict_key_gp_operator]:
                # import funcition from the function py module
                exec(f"from model.genetic.function.gp_operator import {func}")
                if 'data_obj' in eval(func).__annotations__.keys():
                    data_obj_func_list.append(func)
                else:
                    non_data_obj_func_list.append(func)
            func_str = str(formula).replace(")", ",data_obj=data_obj_we_need_to_change)")
            func_str = func_str.replace("(,", "(")
            for str1 in non_data_obj_func_list:
                func_str = func_str.replace(str1 + "(data_obj=data_obj_we_need_to_change)", str1 + "()")

            ###########################################################################
            # all data
            I = getattr(getattr(gp_obj, data_type), "I")
            # I = gp_obj.total_data.I
            res = eval(func_str.replace("data_obj_we_need_to_change", "gp_obj.total_data"))
            if "short" in gp_obj.config[ConstGenetic.config_dict_key_schema]:
                res = res.mul(-1)
        else:
            # function = gp.compile(formula, gp_obj.pset)
            # res = eval(
            #     f"function({','.join([i + '=' + f'gp_obj.{data_type}.' + i for i in gp_obj.config[ConstGenetic.config_dict_key_feature]])})")
            function_str = str(formula)
            for func in self.config[ConstGenetic.config_dict_key_gp_operator]:
                # import funcition from the function py module
                exec(f"from model.genetic.function.gp_operator import {func}")
            for feature_name in self.config[ConstGenetic.config_dict_key_feature]:
                function_str = function_str.replace(feature_name, f"gp_obj.{data_type}.{feature_name}")
            res = eval(function_str)

        res.index = getattr(getattr(gp_obj, data_type), "time_index")
        res.index.name = self.formula
        return res

    def _calculate_factor_weighting_value(self, gp_obj=None, formula=None, data_type='total_data'):
        if gp_obj is None:
            gp_obj = self.gp_obj

        factor_value = self._calculate_factor_value(gp_obj, formula, data_type=data_type)
        weighting_func_str = 'weighting_function_' + gp_obj.config[ConstGenetic.config_dict_key_weighting_function]
        exec(f"from model.genetic.function.weighting_function_set import {weighting_func_str}")
        factor_weighting_value = eval(weighting_func_str)(factor_value)
        return factor_weighting_value

    def _calculate_ret(self, gp_obj=None, formula=None, data_type='total_data'):
        if gp_obj is None:
            gp_obj = self.gp_obj

        factor_weighting_value = self.factor_weighting_value
        pct = getattr(getattr(gp_obj, data_type), "ret")
        pct.index = getattr(getattr(gp_obj, data_type), "time_index")
        ret = factor_weighting_value.shift(2) * pct
        return ret

    def _calculate_strategy_ret(self, gp_obj=None, formula=None, data_type='total_data'):
        if gp_obj is None:
            gp_obj = self.gp_obj

        ret = self._calculate_ret(gp_obj, data_type=data_type)
        order_feature = self.config[ConstGenetic.config_dict_key_strategy_param][
            ConstGenetic.config_dict_value_strategy_param_order_feature]
        strategy_num = self.config[ConstGenetic.config_dict_key_strategy_param][
            ConstGenetic.config_dict_value_strategy_param_key_num]
        feature = getattr(getattr(gp_obj, data_type), order_feature)
        feature = feature.mul(-1).rank(axis=1).applymap(lambda x: 1 if x <= strategy_num else np.nan)
        feature.index = getattr(getattr(gp_obj, data_type), "time_index")
        ret_all = ret.mul(feature).mean(axis=1)
        return ret_all

    def _calculate_strategy_benchmark_ret(self, gp_obj=None, formula=None, data_type='total_data'):
        if gp_obj is None:
            gp_obj = self.gp_obj

        ret = getattr(getattr(gp_obj, data_type), 'ret')
        order_feature = self.config[ConstGenetic.config_dict_key_strategy_param][
            ConstGenetic.config_dict_value_strategy_param_order_feature]
        strategy_num = self.config[ConstGenetic.config_dict_key_strategy_param][
            ConstGenetic.config_dict_value_strategy_param_key_num]
        feature = getattr(getattr(gp_obj, data_type), order_feature)
        feature = feature.mul(-1).rank(axis=1).applymap(lambda x: 1 if x <= strategy_num else np.nan)
        feature.index = getattr(getattr(gp_obj, data_type), "time_index")
        ret_all = ret.mul(feature).mean(axis=1)
        return ret_all

    def _calculate_stats(self, gp_obj=None, formula=None):

        if gp_obj is None:
            gp_obj = self.gp_obj
        total_data_ret = self._calculate_strategy_ret(gp_obj=gp_obj, data_type='total_data')
        try:
            total_data_ret = pd.DataFrame(total_data_ret)
            total_data_ret['Date'] = [datetime.datetime.strftime(pd.to_datetime(total_data_ret.index[x]), "%Y%m%d")
                                      for x in range(len(total_data_ret))]
            max_num = total_data_ret.groupby('Date')['Date'].count().max()
            annual_ret_const = 252 * max_num
            annual_vol_const = 15.8 * np.sqrt(max_num)
        except:
            try:
                annual_ret_const = ConstFut.stats_calculation_constant[gp_obj[ConstGenetic.config_dict_key_frequency]][
                    0]
            except:
                annual_ret_const = 252
            try:
                annual_vol_const = ConstFut.stats_calculation_constant[gp_obj[ConstGenetic.config_dict_key_frequency]][
                    1]
            except:
                annual_vol_const = 15.8

        stats = pd.DataFrame(index=['train', 'test', 'all', 'out_of_sample', 'total'],
                             columns=['Ret', 'IR', 'Calmar', 'Turnover', 'IC_median_abs'])

        def __calculate_annual_return(ret):
            return ret.mean() * annual_ret_const

        def __calculate_annual_ir(ret):
            return ret.mean() / (ret.std() + 0.0001) * annual_vol_const

        def __calculate_calmar(ret):
            mdd = (ret.cumsum() - ret.cumsum().cummax()).min()
            return ret.mean() / (np.abs(mdd) + 0.001) * annual_ret_const

        def __calculate_turnover(res):
            res1 = res.dropna()
            return res1.diff().abs().sum() / len(res1)

        def __calculate_median_abs_ic(__factor_value, __pct):
            return __factor_value.shift(2).reset_index(drop=True).corrwith(__pct.reset_index(drop=True)).abs().median()

        total_data_ret = self._calculate_strategy_ret(gp_obj=gp_obj, data_type='total_data')
        factor_weighting_value = self.factor_weighting_value
        factor_value = self.factor_value
        stats.loc['total', 'Ret'] = __calculate_annual_return(total_data_ret)
        stats.loc['total', 'IR'] = __calculate_annual_ir(total_data_ret)
        stats.loc['total', 'Calmar'] = __calculate_calmar(total_data_ret)
        stats.loc['total', 'Turnover'] = __calculate_turnover(factor_weighting_value).mean()
        stats.loc['total', 'IC_median_abs'] = __calculate_median_abs_ic(factor_value,
                                                                        getattr(getattr(gp_obj, 'total_data'), "ret"))

        for item in ['train', 'test', 'all', 'out_of_sample']:
            time_index_list = getattr(getattr(gp_obj, item + '_data'), 'time_index')
            ret_sub_data = total_data_ret[total_data_ret.index.isin(time_index_list)]
            factor_weighting_sub_data = factor_weighting_value[factor_weighting_value.index.isin(time_index_list)]
            factor_sub_data = factor_value[factor_value.index.isin(time_index_list)]
            stats.loc[item, 'Ret'] = __calculate_annual_return(ret_sub_data)
            stats.loc[item, 'IR'] = __calculate_annual_ir(ret_sub_data)
            stats.loc[item, 'Calmar'] = __calculate_calmar(ret_sub_data)
            stats.loc[item, 'Turnover'] = __calculate_turnover(factor_weighting_sub_data).mean()
            stats.loc[item, 'IC_median_abs'] = __calculate_median_abs_ic(factor_sub_data,
                                                                         getattr(getattr(gp_obj, item + '_data'),
                                                                                 "ret"))
        return stats

    def plot_strategy_nv_and_stats(self):
        import matplotlib.pyplot as plt
        stats_show = self.stats.copy()
        for i in range(len(stats_show)):
            stats_show.iloc[i, 0] = str(round(stats_show.iloc[i, 0] * 100, 2)) + '%'
            stats_show.iloc[i, 1] = str(round(stats_show.iloc[i, 1], 2))
            stats_show.iloc[i, 2] = str(round(stats_show.iloc[i, 2], 2))
            stats_show.iloc[i, 3] = str(round(stats_show.iloc[i, 3] * 100, 2)) + '%'
            stats_show.iloc[i, 4] = str(round(stats_show.iloc[i, 4], 4))

        fig, axs = plt.subplots(3, 1, figsize=(20, 30))

        axs[0].axis('off')
        the_table = axs[0].table(cellText=stats_show.values, colLabels=stats_show.columns, rowLabels=stats_show.index,
                                 cellLoc='center', loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(20)
        the_table.scale(1, 4)

        axs[1].plot(self.strategy_nv.fillna(method='ffill').fillna(0), label='strategy')

        axs[1].plot(self.strategy_benchmark_nv, label='benchmark')
        axs[1].legend()
        axs[1].set_title("strategy nv and undetlying nv")

        axs[2].plot(self.strategy_benchmark_nv, color='white')
        axs[2].plot(self.strategy_nv - self.strategy_nv.cummax(), lw=3, label='drawdown', alpha=0.5)
        axs[2].fill_between(x=self.strategy_nv.index, y1=np.array([0] * len(self.strategy_nv)),
                            y2=self.strategy_nv - self.strategy_nv.cummax(), alpha=0.7)
        axs[2].legend()
        axs[2] = axs[2].twinx()
        axs[2].plot(
            self._calculate_strategy_ret().rolling(60).mean() / self._calculate_strategy_ret().rolling(60).std(),
            label='rolling sharpe', color='red', alpha=0.5)
        axs[2].legend()
        axs[2].set_title('Drawdown and Rolling Sharpe')

    def plot_sub_fund_nv(self):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.sub_nv.fillna(method='ffill').fillna(0), label=self.sub_nv.columns)
        ax.legend()
        ax.set_title('sub_nv')

    def save_to_pickle(self, file_location):
        import pickle
        with open(file_location, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def __name_dict(self):
        data = pd.DataFrame()

        for k, v in self.config.items():
            data.loc[0, k] = str(v)
        # split method
        data.loc[0, 'train_test_split_method'] = self.config[ConstGenetic.config_dict_key_train_test_split_method][
            'method']
        # population
        data.loc[0, ConstGenetic.config_dict_key_gp_run_param_key_population_num] = \
            self.config[ConstGenetic.config_dict_key_gp_run_param][
                ConstGenetic.config_dict_key_gp_run_param_key_population_num]
        # ngen
        data.loc[0, ConstGenetic.config_dict_key_gp_run_param_key_ngen] = \
            self.config[ConstGenetic.config_dict_key_gp_run_param][ConstGenetic.config_dict_key_gp_run_param_key_ngen]
        # max depth and min depth
        data.loc[0, ConstGenetic.config_dict_key_gp_run_param_key_expression_max_depth] = \
            self.config[ConstGenetic.config_dict_key_gp_run_param][
                ConstGenetic.config_dict_key_gp_run_param_key_expression_max_depth]
        data.loc[0, ConstGenetic.config_dict_key_gp_run_param_key_expression_min_depth] = \
            self.config[ConstGenetic.config_dict_key_gp_run_param][
                ConstGenetic.config_dict_key_gp_run_param_key_expression_min_depth]

        # mate and mutate depth
        data.loc[0, ConstGenetic.config_dict_values_toolbox_key_mate_max] = \
            self.config[ConstGenetic.config_dict_key_toolbox][ConstGenetic.config_dict_values_toolbox_key_mate_max]
        data.loc[0, ConstGenetic.config_dict_values_toolbox_key_mutate_max] = \
            self.config[ConstGenetic.config_dict_key_toolbox][ConstGenetic.config_dict_values_toolbox_key_mutate_max]

        # strategy param num and order feature
        data.loc[0, ConstGenetic.config_dict_value_strategy_param_key_num] = \
            self.config[ConstGenetic.config_dict_key_strategy_param][
                ConstGenetic.config_dict_value_strategy_param_key_num]
        data.loc[0, ConstGenetic.config_dict_value_strategy_param_order_feature] = \
            self.config[ConstGenetic.config_dict_key_strategy_param][
                ConstGenetic.config_dict_value_strategy_param_order_feature]

        for i in range(len(self.stats)):
            for j in range(len(self.stats.columns)):
                data.loc[0, self.stats.columns[j] + '_' + self.stats.index[i]] = self.stats.iloc[i, j]

        data.loc[0, "Formula"] = self.formula
        data.loc[0, 'run_time'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m%d_%H%M%S_%f')
        return data


def result_save_function(gp_obj, result):
    if len(result) == 0:
        return
    else:
        date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m%d_%H%M%S_%f')
        data_save_local = pd.DataFrame()

        formula = result[0]
        gp_result = GeneticProgrammingResult(gp_obj.config, formula)
        data = gp_result.name_dict
        s = ""
        for a in ConstGenetic.select_column:
            s = s + str(data.loc[0, a]) + '_'
        s = s.replace(".0", "").replace("-", "").replace("{'method': '", "").replace("'}", "")
        s = s.replace("cm_cn_group_", "").replace("pattern_recognition", "pr")
        pickle_save_path = ConstPath.single_factor_result_output_pickle_path + s + date_str + '\\'
        if len(pickle_save_path) > 220:
            pickle_save_path = ConstPath.single_factor_result_output_pickle_path + date_str + '\\'
        for i in range(len(result)):
            formula = result[i]
            gp_result = GeneticProgrammingResult(gp_obj.config, formula)
            data = gp_result.name_dict

            pickle_save_location = pickle_save_path + 'Factor_' + str(i) + '.pkl'
            if not os.path.exists(pickle_save_path):
                try:
                    os.mkdir(pickle_save_path)
                except Exception as e:
                    os.makedirs(pickle_save_path)
            gp_result.save_to_pickle(file_location=pickle_save_location)

            data.loc[0, 'save_location'] = pickle_save_location
            data.loc[0, 'save_location_time_str'] = date_str

            data_save_local = pd.concat([data_save_local, data])
            data_save_local.index = range(len(data_save_local))
            data_save_local.to_csv(pickle_save_path + 'total.csv')

    # index_loop = True
    # while index_loop:
    #     try:
    #         try:
    #             with open(ConstPath.summary_all_result_mapping_dict_pickle_location, 'rb') as file_all:
    #                 data_all = pickle.load(file_all)
    #         except FileNotFoundError:
    #             data_all = pd.DataFrame()
    #         data_all = pd.concat([data_all, data])
    #         data_all.index = range(len(data_all))
    #         index_loop = False
    #     except:
    #         time.sleep(2)
    #
    # index_loop = True
    # while index_loop:
    #     try:
    #         with open(ConstPath.summary_all_result_mapping_dict_pickle_location, 'wb') as file_all:
    #             pickle.dump(data_all, file_all, pickle.HIGHEST_PROTOCOL)
    #         index_loop = False
    #     except:
    #         time.sleep(2)

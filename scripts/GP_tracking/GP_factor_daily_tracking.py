import sys

sys.path.append('..')
import model.genetic.GP as GP
import model.genetic.config_list as Config_List
import datetime
from deap import tools
import time
import random

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
from model.genetic.function.result_save_function import GeneticProgrammingResult
import scripts.GP_tracking.util.crypto_future_hourly_data_update as UpdateData
import json

save_path = f"C:\\Users\\jason.huang\\research\\scripts_working\\GP_tracking\\crypto\\crypto_binance_future_hourly\\factor_value\\"


def create_save_file_path(end_date=None):
    if end_date is None:
        end_date = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")
    if not os.path.exists(f"{save_path}{end_date}\\factor_value\\"):
        try:
            os.mkdir(f"{save_path}{end_date}\\factor_value\\")
        except:
            try:
                os.makedirs(f"{save_path}{end_date}\\factor_value\\")
            except Exception as e:
                print(e)
    if not os.path.exists(f"{save_path}{end_date}\\factor_weight\\"):
        try:
            os.mkdir(f"{save_path}{end_date}\\factor_weight\\")
        except:
            try:
                os.makedirs(f"{save_path}{end_date}\\factor_weight\\")
            except Exception as e:
                print(e)
    if not os.path.exists(f"{save_path}{end_date}\\factor_ret\\"):
        try:
            os.mkdir(f"{save_path}{end_date}\\factor_ret\\")
        except:
            try:
                os.makedirs(f"{save_path}{end_date}\\factor_ret\\")
            except Exception as e:
                print(e)


def read_factor_list(file_location):
    factor_list_data = pd.read_csv(file_location)
    return factor_list_data


def update_data(json_file_location):
    with open(json_file_location, 'r') as f:
        config_dict = json.load(f)
    UpdateData.update_crypto_future_hourly_feature_data(tickers_list=None, start_date=config_dict.setdefault(
        'config_dict_value_default_start_date', None), end_date=None, table_name=None, key_word_list=None,
                                                        feature_save_path=config_dict.setdefault(
                                                            'input_data_feature_path', None))


def update_config(config_dict, json_file_location):
    with open(json_file_location, 'r') as f:
        config_dict_revise = json.load(f)
    try:
        config_dict[ConstGenetic.config_dict_key_feature] = config_dict_revise[
            'config_dict_value_default_feature_factor']
    except:
        pass
    try:
        config_dict[ConstGenetic.config_dict_key_start_date] = config_dict_revise[
            'config_dict_value_default_start_date']
    except:
        pass
    try:
        config_dict[ConstGenetic.config_dict_key_end_date] = config_dict_revise[
            'config_dict_value_default_end_date']
    except:
        pass
    try:
        config_dict[ConstGenetic.config_dict_key_out_of_sample_date] = config_dict_revise[
            'config_dict_value_default_out_of_sample_date']
    except:
        pass
    try:
        config_dict[ConstGenetic.config_dict_key_frequency] = config_dict_revise[
            'config_dict_value_default_frequency']
    except:
        pass
    try:
        config_dict['input_data_feature_path'] = config_dict_revise['input_data_feature_path']
        config_dict['input_data_return_pct_path'] = config_dict_revise['input_data_return_pct_path']
    except:
        pass
    return config_dict


def update_factor_value_and_weight(factor_file_location, json_file_location, update_date):
    if update_date is None:
        update_date = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")
    create_save_file_path(end_date=update_date)
    factor_data = read_factor_list(file_location=factor_file_location)

    for i in range(len(factor_data)):
        try:
            formula_string = factor_data.iloc[i, 0]
            weight_function = factor_data.iloc[i, 1]
            factor_name_str = factor_data.iloc[i, -1]
            config_dict = Config_List.config_naming_func(strategy_name='crypto_future',
                                                         tickers_group_name='crypto_binance_future_all',
                                                         weighting_function=weight_function, population_num=int('100'),
                                                         max_depth=int('2'), ngen=int('1'))
            config_dict = update_config(config_dict, json_file_location)
            gp_result = GeneticProgrammingResult(config_dict, formula_string)
            factor_value = gp_result.factor_value
            factor_value.to_csv(f"{save_path}{update_date}\\factor_value\\{factor_name_str}.csv")
            print(f"{factor_name_str}: factor value is ok")
            weight_value = gp_result.factor_weighting_value

            weight_value.to_csv(f"{save_path}{update_date}\\factor_weight\\{factor_name_str}.csv")
            print(f"{factor_name_str}: factor weight is ok")
            ret_value = gp_result.sub_nv.diff()
            ret_value.to_csv(f"{save_path}{update_date}\\factor_ret\\{factor_name_str}.csv")
            print(f"{factor_name_str}: factor ret is ok")
            print(f"{i}/{len(factor_data)} is ok")
        except Exception as e:
            print(f"{i}/{len(factor_data)} is not ok  Reason is {e}")


if __name__ == '__main__':
    json_file_location = "C:\\Users\\jason.huang\\research\\scripts_working\\GP_tracking\\crypto\\crypto_binance_future_hourly\\config.json"
    factor_file_location = "C:\\Users\\jason.huang\\research\\scripts_working\\GP_tracking\\crypto\\crypto_binance_future_hourly\\factor_name_list\\crypto_binance_hourly_factor_20221206.csv"
    update_data(json_file_location)
    # factor_list = read_factor_list(file_location=factor_file_location)
    update_factor_value_and_weight(factor_file_location=factor_file_location, json_file_location=json_file_location,
                                   update_date=None)

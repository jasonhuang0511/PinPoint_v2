import numpy as np
import pandas as pd

from model.genetic.GP import DataObj
import model.constants.genetic as ConstGenetic

"""
Fitness Function Set:

    a set of fitness function (evaluate function) in the genetic programming
    
    called by config_dict["FitnessFunction"] in model.genetic.config.py
    
    ** Structure: input should be the same:
        1. res: pd.DataFrame without index,  result of the genetic formula
        2. pct: pd.DataFrame without index,  pct matrix of the assets
        3. data_obj: DataObj, a class which saved the feature information such as origin data and origins 
    
    ** Principle:
        default set of Genetic Programming is max the statistics result of fitness function
        we want to construct the fitness function to meet the requirement: larger fitness function result, better result
"""


####################################
#
# fitness function structure
#
####################################

def fitness_function_structure(factor_value: pd.DataFrame = pd.DataFrame(),
                               pct: pd.DataFrame = pd.DataFrame(),
                               data_obj: DataObj = DataObj(), config_dict: dict = None) -> float:
    max_indicator_func = config_dict[ConstGenetic.config_dict_key_fitness_max_stats_function]
    exec(f"from model.genetic.function.fitness_max_stats_function_set import {max_indicator_func}")
    if 'factor_ic' in max_indicator_func:
        stats = eval(max_indicator_func)(factor_value, pct)
    elif 'strategy_nv_indicator' in max_indicator_func:

        weighting_function_str = 'weighting_function_' + config_dict[ConstGenetic.config_dict_key_weighting_function]
        exec(f"from model.genetic.function.weighting_function_set import {weighting_function_str}")
        res = eval(weighting_function_str)(factor_value)

        strategy_num = config_dict[ConstGenetic.config_dict_key_strategy_param][
            ConstGenetic.config_dict_value_strategy_param_key_num]
        order_feature = config_dict[ConstGenetic.config_dict_key_strategy_param][
            ConstGenetic.config_dict_value_strategy_param_order_feature]

        ret = res.shift(2) * pct

        if strategy_num >= len(pct.columns):
            ret_all = ret.mean(axis=1)
        else:
            feature = getattr(data_obj, order_feature)
            feature = feature.mul(-1).rank(axis=1).applymap(lambda x: 1 if x <= strategy_num else np.nan)
            ret_all = ret.mul(feature).mean(axis=1)

        stats = eval(max_indicator_func)(ret_all)

    else:
        stats = -np.inf
    return stats

#
# ############################################################
# #
# #   factor fitness function
# #
# ############################################################
# def factor_max_sum_abs_ic(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                           data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max sum(abs(IC(return,factor))) of assets
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: sum(abs(ic of assets returns and factors))
#     """
#     res1 = res.replace([np.inf, -np.inf], np.nan)
#     res1 = res1.where(pct.isnull() == False).dropna(axis=0, how='all')
#
#     cor = res1.corrwith(pct, axis=0).dropna()
#
#     try:
#         result = cor.abs().sum()
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# def factor_max_sum_ic(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                       data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max sum(IC(return,factor)) of assets
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: sum(ic of assets returns and factors)
#     """
#     res1 = res.replace([np.inf, -np.inf], np.nan)
#     res1 = res1.where(pct.isnull() == False).dropna(axis=0, how='all')
#
#     cor = res1.corrwith(pct, axis=0).dropna()
#
#     try:
#         result = cor.sum()
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# def factor_simple_long_short_max_calmar(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                         data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max sum(IC(return,factor)) of assets
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: sum(ic of assets returns and factors)
#     """
#     res1 = res.replace([np.inf, -np.inf, np.nan], 0)
#     ret = res1.applymap(np.sign).shift(2) * pct
#     ret_mean = ret.mean(axis=1)
#     nv = ret.mean(axis=1).cumsum()
#     mdd = (nv - nv.cummax()).min()
#     try:
#         result = ret_mean.sum() / (np.abs(mdd) + 0.0001)
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# def factor_weight_function_phi_max_calmar(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                           data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max sum(IC(return,factor)) of assets
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: sum(ic of assets returns and factors)
#     """
#     res1 = res.replace([np.inf, -np.inf, np.nan], 0)
#     ret = res1.applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2) * pct
#     ret_mean = ret.mean(axis=1)
#     nv = ret.mean(axis=1).cumsum()
#     mdd = (nv - nv.cummax()).min()
#     try:
#         result = ret_mean.sum() / (np.abs(mdd) + 0.0001)
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# def factor_weight_function_phi_max_ir(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                       data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max sum(IC(return,factor)) of assets
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: sum(ic of assets returns and factors)
#     """
#     if (res.applymap(lambda x: 1 if np.isnan(x) else 0).sum() / len(res)).max() > 0.25:
#         return -np.inf
#     res1 = res.replace([np.inf, -np.inf, np.nan], 0)
#     ret = res1.applymap(lambda x: x * np.exp(-x * x / 4) / 0.89).shift(2) * pct
#     ret_mean = ret.mean(axis=1)
#     nv = ret.mean(axis=1).cumsum()
#     try:
#         result = ret_mean.mean() / (ret_mean.std() + 0.0001)
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# ############################################################
# #
# #   pattern fitness function
# #
# ############################################################
# def pattern_recognition_max_ir_ret(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                    data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy return + strategy IR
#     strategy: at the same time, pick up at most 10 assets (sort by volumes) by genetic result
#             if qualified assets number is larger than 10, sort by volumes then pick up 10 assets with most large volumes
#             if qualified assets number is less than 10, pick up all the qualified assets with each weigh 10%
#
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: strategy return + strategy IR
#     """
#     volume = getattr(data_obj, "volume")
#     res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
#     volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)
#     ret_cond = pct.where(volumerank <= 10)
#     ret_mean = ret_cond.sum(axis=1) / 10
#     if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
#         return -np.inf
#     else:
#         return ret_mean.dropna().mean() / ret_mean.dropna().std() + ret_mean.fillna(0).sum()
#
#
# def pattern_recognition_stock_index_max_ir_plus_ret(res: pd.DataFrame = pd.DataFrame(),
#                                                     pct: pd.DataFrame = pd.DataFrame(),
#                                                     data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy return + strategy IR of stock index future
#         strategy: 3 assets with equal weight
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: strategy return + strategy IR
#     """
#     if (res.applymap(lambda x: 1 if np.isnan(x) else 0).sum() / len(res)).max() > 0.25:
#         return -np.inf
#     res1 = res.replace([np.inf, -np.inf, np.nan], 0)
#     ret = res1.shift(2) * pct
#     ret_mean = ret.mean(axis=1)
#     try:
#         result = ret_mean.mean() / (ret_mean.std() + 0.0001)
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# def pattern_recognition_stock_index_max_ret(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                             data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy return  of stock index future
#         strategy: 3 assets with equal weight
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: strategy return
#     """
#     volume = getattr(data_obj, "volume")
#     res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
#     volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)
#     ret_cond = pct.where(volumerank <= len(volume.columns))
#     ret_mean = ret_cond.sum(axis=1) / len(volume.columns)
#     if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
#         return -np.inf
#     else:
#         return ret_mean.dropna().mean()
#
#
# def pattern_recognition_stock_index_max_ir(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                            data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy IR  of stock index future
#         strategy: 3 assets with equal weight
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return:  strategy IR
#     """
#     volume = getattr(data_obj, "volume")
#     res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
#     volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)
#     ret_cond = pct.where(volumerank <= len(volume.columns))
#     ret_mean = ret_cond.sum(axis=1) / len(volume.columns)
#     if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
#         return -np.inf
#     else:
#         return ret_mean.dropna().mean() / ret_mean.dropna().std()
#
#
# def pattern_recognition_stock_index_max_ir_short(res: pd.DataFrame = pd.DataFrame(), pct: pd.DataFrame = pd.DataFrame(),
#                                                  data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy IR  of stock index future
#         strategy: 3 assets with equal weight
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return:  strategy IR
#     """
#     volume = getattr(data_obj, "volume")
#     res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
#     volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)
#     ret_cond = pct.where(volumerank <= len(volume.columns))
#     ret_mean = ret_cond.sum(axis=1) / len(volume.columns) * (-1)
#     if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
#         return -np.inf
#     else:
#         return ret_mean.dropna().mean() / ret_mean.dropna().std()
#
#
# def pattern_recognition_stock_index_max_ir_plus_ret_short(res: pd.DataFrame = pd.DataFrame(),
#                                                           pct: pd.DataFrame = pd.DataFrame(),
#                                                           data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy return + strategy IR of stock index future
#         strategy: 3 assets with equal weight
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: strategy return + strategy IR
#     """
#     # volume = getattr(data_obj, "volume")
#     # res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
#     # volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)
#     # ret_cond = pct.where(volumerank <= len(volume.columns))
#     # ret_mean = ret_cond.sum(axis=1) / len(volume.columns) * (-1)
#     # if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
#     #     return -np.inf
#     # else:
#     #     return ret_mean.dropna().mean() / (ret_mean.dropna().std() + 0.01) + ret_mean.fillna(0).sum()
#     #
#
#     res1 = res.replace([np.inf, -np.inf, np.nan], 0)
#     ret = res1.shift(2) * pct * (-1)
#     ret_mean = ret.mean(axis=1)
#     try:
#         result = ret_mean.mean() / (ret_mean.std() + 0.0001) * 15.8 + ret_mean.mean() * 252
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf
#
#
# def pattern_recognition_stock_index_max_ir_plus_ret_long(res: pd.DataFrame = pd.DataFrame(),
#                                                          pct: pd.DataFrame = pd.DataFrame(),
#                                                          data_obj: DataObj = DataObj()) -> pd.DataFrame:
#     """
#     Max strategy return + strategy IR of stock index future
#         strategy: 3 assets with equal weight
#     :param res: result of genetic formula
#     :param pct: pct matrix of assets
#     :param data_obj: train data feature
#     :return: strategy return + strategy IR
#     """
#     # volume = getattr(data_obj, "volume")
#     # res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
#     # volumerank = volume.mul(-1).where(res1 > 0).rank(axis=1)
#     # ret_cond = pct.where(volumerank <= len(volume.columns))
#     # ret_mean = ret_cond.sum(axis=1) / len(volume.columns) * (-1)
#     # if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
#     #     return -np.inf
#     # else:
#     #     return ret_mean.dropna().mean() / (ret_mean.dropna().std() + 0.01) + ret_mean.fillna(0).sum()
#     #
#
#     res1 = res.replace([np.inf, -np.inf, np.nan], 0)
#     ret = res1.shift(2) * pct
#     ret_mean = ret.mean(axis=1)
#     try:
#         result = ret_mean.mean() / (ret_mean.std() + 0.0001) * 15.8 + ret_mean.mean() * 252
#         if result < np.inf:
#             return result
#         else:
#             return -np.inf
#     except Exception as e:
#         return -np.inf


#
# ############################################################
# #
# #   weighting function
# #
# ############################################################
# def weighting_function_simple_long_short(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
#     return factor.applymap(np.sign)
#
#
# def weighting_function_phi_function(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
#     return factor.applymap(lambda x: x * np.exp(-x * x / 4) / 0.89)
#
#
# def weighting_function_tanh(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
#     return factor.applymap(np.tanh)
#
#
# def weighting_function_sigmoid(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
#     return factor.applymap(lambda x: 1 / (1 + np.exp(-1 * x)))
#
#
# def weighting_function_identity(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
#     return factor.applymap(lambda x: x)
#
#
# ########################################
# #
# #   fitness_max_stats_function
# #
# ########################################
#
# def strategy_nv_indicator_annual_ret(ret):
#     return ret.mean() * 252
#
#
# def strategy_nv_indicator_annual_ir(ret):
#     return ret.mean() / (ret.std() + 0.0001) * 15.8
#
#
# def strategy_nv_indicator_annual_calmar(ret):
#     mdd = (ret.cumsum() - ret.cumsum().cummax()).min()
#     return ret.sum() / (mdd + 0.0001)
#
#
# def factor_ic_sum_abs(factor, pct):
#     return factor.shift(2).corrwith(pct).sum().abs()

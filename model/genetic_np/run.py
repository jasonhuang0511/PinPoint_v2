import sys

sys.path.append('..')
sys.path.append('C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2')
import os
import time
from multiprocessing import Pool
import random
import itertools
from scripts import LoggingConfig

logger = LoggingConfig.Config().get_config()

param_list = ["-sn", "-t", "-sm", "-sc", "-wf", "-ind", "-ngen", "-dep", "-pop"]


def run_func(str_code_param):
    logger.info("process random waiting time before gp")
    time.sleep(random.random() * 15)
    os.system(
        f"python  C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\model\\genetic\\gp_main.py {str_code_param}")
    logger.info("process random waiting time after gp")
    time.sleep(random.random() * 15)


def generate_config_list(loop=20):
    ###################################
    #
    # daily stock index andinterest rate
    #
    #####################################
    # ts factor
    strategy_name_list = ['ts_factor']
    tickers_group_name_list = ['cm_cn_group_stock_index', 'cm_cn_group_interest_rate']
    split_method_dict_list = ['every_other_row', 'ts_split']
    gp_schema_list = ['ts_factor']
    weighting_function_list = ['simple_long_short', 'phi_function', 'tanh', 'sigmoid']
    fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
                                       'strategy_nv_indicator_annual_ir_plus_ret',
                                       'strategy_nv_indicator_annual_calmar']
    ngen_list = [2, 3, 4, 5, 6]
    max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    population_num_list = [2000, 3000, 5000, 10000, 20000]

    code_param_list = []
    for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
                                   weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
                                   population_num_list):
        code_param_list.append(list(items))
    code_str_param_list_ts_factor = []
    for i in range(len(code_param_list)):
        str_list = []
        for j in range(len(param_list)):
            str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
        code_str_param_list_ts_factor.append(" ".join(str_list))

    # #######################################################
    # # pr long
    # strategy_name_list = ['pr_long']
    # tickers_group_name_list = ['cm_cn_group_stock_index', 'cm_cn_group_interest_rate']
    # split_method_dict_list = ['every_other_row', 'ts_split']
    # gp_schema_list = ['pattern_recognition_long']
    #
    # weighting_function_list = ['identity']
    # fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
    #                                    'strategy_nv_indicator_annual_ir_plus_ret',
    #                                    'strategy_nv_indicator_annual_calmar']
    # ngen_list = [2, 3, 4, 5, 6]
    # max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    # population_num_list = [2000, 3000, 5000, 10000, 20000]
    #
    # code_param_list = []
    # for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
    #                                weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
    #                                population_num_list):
    #     code_param_list.append(list(items))
    # code_str_param_list_pr_long = []
    # for i in range(len(code_param_list)):
    #     str_list = []
    #     for j in range(len(param_list)):
    #         str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
    #     code_str_param_list_pr_long.append(" ".join(str_list))
    #
    # ############################################################
    # # pr short
    # strategy_name_list = ['pr_short']
    # tickers_group_name_list = ['cm_cn_group_stock_index', 'cm_cn_group_interest_rate']
    # split_method_dict_list = ['every_other_row', 'ts_split']
    # gp_schema_list = ['pattern_recognition_short']
    #
    # weighting_function_list = ['identity']
    # fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
    #                                    'strategy_nv_indicator_annual_ir_plus_ret',
    #                                    'strategy_nv_indicator_annual_calmar']
    # ngen_list = [2, 3, 4, 5, 6]
    # max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    # population_num_list = [2000, 3000, 5000, 10000, 20000]
    #
    # code_param_list = []
    # for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
    #                                weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
    #                                population_num_list):
    #     code_param_list.append(list(items))
    # code_str_param_list_pr_short = []
    # for i in range(len(code_param_list)):
    #     str_list = []
    #     for j in range(len(param_list)):
    #         str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
    #     code_str_param_list_pr_short.append(" ".join(str_list))
    # code_str_param_list = code_str_param_list_ts_factor + code_str_param_list_pr_long + code_str_param_list_pr_short
    # return code_str_param_list * loop

    #####################################
    # ts factor
    # strategy_name_list = ['intraday_future_ts_factor']
    # tickers_group_name_list = ['cm_cn_group_stock_index', 'cm_cn_group_interest_rate', 'cm_cn_group_selected_1',
    #                            'cm_cn_group_selected_2', 'cm_cn_group_selected_3', 'cm_cn_group_selected_4']
    # split_method_dict_list = ['every_other_row', 'ts_split']
    # gp_schema_list = ['ts_factor']
    # weighting_function_list = ['simple_long_short', 'phi_function', 'tanh', 'sigmoid']
    # fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
    #                                    'strategy_nv_indicator_annual_ir_plus_ret',
    #                                    'strategy_nv_indicator_annual_calmar']
    # ngen_list = [2, 3, 4, 5, 6]
    # max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    # population_num_list = [2000, 3000, 5000, 10000, 20000]
    #
    # code_param_list = []
    # for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
    #                                weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
    #                                population_num_list):
    #     code_param_list.append(list(items))
    # code_str_param_list_ts_factor = []
    # for i in range(len(code_param_list)):
    #     str_list = []
    #     for j in range(len(param_list)):
    #         str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
    #     code_str_param_list_ts_factor.append(" ".join(str_list))
    #
    # #######################################################
    # # pr long
    # strategy_name_list = ['intraday_future_pr_long']
    # tickers_group_name_list = ['cm_cn_group_stock_index', 'cm_cn_group_interest_rate', 'cm_cn_group_selected_1',
    #                            'cm_cn_group_selected_2', 'cm_cn_group_selected_3', 'cm_cn_group_selected_4',
    #                            'cm_cn_group_selected_5']
    # split_method_dict_list = ['every_other_row', 'ts_split']
    # gp_schema_list = ['pattern_recognition_long']
    #
    # weighting_function_list = ['identity']
    # fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
    #                                    'strategy_nv_indicator_annual_ir_plus_ret',
    #                                    'strategy_nv_indicator_annual_calmar']
    # ngen_list = [2, 3, 4, 5, 6]
    # max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    # population_num_list = [2000, 3000, 5000, 10000, 20000]
    #
    # code_param_list = []
    # for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
    #                                weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
    #                                population_num_list):
    #     code_param_list.append(list(items))
    # code_str_param_list_pr_long = []
    # for i in range(len(code_param_list)):
    #     str_list = []
    #     for j in range(len(param_list)):
    #         str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
    #     code_str_param_list_pr_long.append(" ".join(str_list))
    #
    # ############################################################
    # # pr short
    # strategy_name_list = ['intraday_future_pr_short']
    # tickers_group_name_list = ['cm_cn_group_stock_index', 'cm_cn_group_interest_rate', 'cm_cn_group_selected_1',
    #                            'cm_cn_group_selected_2', 'cm_cn_group_selected_3', 'cm_cn_group_selected_4',
    #                            'cm_cn_group_selected_5']
    # split_method_dict_list = ['every_other_row', 'ts_split']
    # gp_schema_list = ['pattern_recognition_short']
    # weighting_function_list = ['identity']
    # fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
    #                                    'strategy_nv_indicator_annual_ir_plus_ret',
    #                                    'strategy_nv_indicator_annual_calmar']
    # ngen_list = [2, 3, 4, 5, 6]
    # max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    # population_num_list = [2000, 3000, 5000, 10000, 20000]
    #
    # code_param_list = []
    # for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
    #                                weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
    #                                population_num_list):
    #     code_param_list.append(list(items))
    # code_str_param_list_pr_short = []
    # for i in range(len(code_param_list)):
    #     str_list = []
    #     for j in range(len(param_list)):
    #         str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
    #     code_str_param_list_pr_short.append(" ".join(str_list))
    # code_str_param_list = code_str_param_list_ts_factor + code_str_param_list_pr_long + code_str_param_list_pr_short

    #########################################
    #
    #   hourly data of crypto
    #  run at 2022-11-14
    #
    ########################################

    strategy_name_list = ['crypto']
    tickers_group_name_list = ['crypto_binance_all', 'crypto_binance_selected']
    split_method_dict_list = ['every_other_row', 'ts_split']
    gp_schema_list = ['ts_factor']
    weighting_function_list = ['simple_long_short', 'phi_function', 'tanh', 'sigmoid']
    fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
                                       'strategy_nv_indicator_annual_ir_plus_ret',
                                       'strategy_nv_indicator_annual_calmar']
    ngen_list = [2, 3, 4, 5, 6]
    max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    population_num_list = [2000, 3000, 5000, 10000, 20000]

    code_param_list = []
    for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
                                   weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
                                   population_num_list):
        code_param_list.append(list(items))
    code_str_param_list_ts_factor = []
    for i in range(len(code_param_list)):
        str_list = []
        for j in range(len(param_list)):
            str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
        code_str_param_list_ts_factor.append(" ".join(str_list))

    #######################################################
    # pr long
    strategy_name_list = ['crypto']
    tickers_group_name_list = ['crypto_binance_all', 'crypto_binance_selected']
    split_method_dict_list = ['every_other_row', 'ts_split']
    gp_schema_list = ['pattern_recognition_long']

    weighting_function_list = ['identity']
    fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
                                       'strategy_nv_indicator_annual_ir_plus_ret',
                                       'strategy_nv_indicator_annual_calmar']
    ngen_list = [2, 3, 4, 5, 6]
    max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    population_num_list = [2000, 3000, 5000, 10000, 20000]

    code_param_list = []
    for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
                                   weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
                                   population_num_list):
        code_param_list.append(list(items))
    code_str_param_list_pr_long = []
    for i in range(len(code_param_list)):
        str_list = []
        for j in range(len(param_list)):
            str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
        code_str_param_list_pr_long.append(" ".join(str_list))

    ############################################################
    # pr short
    strategy_name_list = ['crypto']
    tickers_group_name_list = ['crypto_binance_all', 'crypto_binance_selected']
    split_method_dict_list = ['every_other_row', 'ts_split']
    gp_schema_list = ['pattern_recognition_short']
    weighting_function_list = ['identity']
    fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
                                       'strategy_nv_indicator_annual_ir_plus_ret',
                                       'strategy_nv_indicator_annual_calmar']
    ngen_list = [2, 3, 4, 5, 6]
    max_depth_list = [5, 6, 7, 8, 9, 10, 20]
    population_num_list = [2000, 3000, 5000, 10000, 20000]

    code_param_list = []
    for items in itertools.product(strategy_name_list, tickers_group_name_list, split_method_dict_list, gp_schema_list,
                                   weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
                                   population_num_list):
        code_param_list.append(list(items))
    code_str_param_list_pr_short = []
    for i in range(len(code_param_list)):
        str_list = []
        for j in range(len(param_list)):
            str_list.append(param_list[j] + ' ' + str(code_param_list[i][j]))
        code_str_param_list_pr_short.append(" ".join(str_list))
    code_str_param_list = code_str_param_list_ts_factor + code_str_param_list_pr_long + code_str_param_list_pr_short

    return code_str_param_list * loop


if __name__ == '__main__':
    logger.info("start gp")
    code_str_list = generate_config_list(loop=30)
    logger.info(f"total: {len(code_str_list)} gp tasks")
    pool = Pool(2)
    pool.map(run_func, code_str_list)
    logger.info("end gp")

    # for i in range(50):
    #     run_func(i)

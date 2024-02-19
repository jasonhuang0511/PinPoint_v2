tickers_group_name_list = ['cm_cn_group_grains_oilseeds', 'cm_cn_group_livestock', 'cm_cn_group_softs',
                           'cm_cn_group_base_metal', 'cm_cn_group_black', 'cm_cn_group_chemicals',
                           'cm_cn_group_energy',
                           'cm_cn_group_stock_index', 'cm_cn_group_interest_rate', 'cm_cn_sector_agriculture',
                           'cm_cn_sector_industrials', 'cm_cn_sector_refineries', 'cm_cn_sector_financial',
                           'cm_cn_sector_commodity', 'cm_cn_all']
data_freq_list = ['1D']
ret_type_list = ['close']
split_method_dict_list = [{'method': 'every_other_row'}, {'method': 'ts_split', 'ratio': 0.8}]
gp_schema_list = ['factor']
weighting_function_list = ['simple_long_short', 'phi_function', 'tanh', 'sigmoid', 'identity']
fitness_max_stats_function_list = ['strategy_nv_indicator_annual_ret', 'strategy_nv_indicator_annual_ir',
                                   'strategy_nv_indicator_annual_ir_plus_ret',
                                   'strategy_nv_indicator_annual_calmar', 'factor_ic_sum_abs']
ngen_list = [2, 3, 4, 5, 6]
max_depth_list = [5, 6, 7, 8, 9, 10, 20]
population_num_list = [2000, 3000, 5000, 10000, 20000]

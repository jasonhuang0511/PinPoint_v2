import sys

sys.path.append('..')
sys.path.append('C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2')

import model.constants.futures as ConstFut
import model.constants.genetic as ConstGenetic


def config_naming_func(strategy_name=None, tickers_group_name=None, start_date=None, end_date=None,
                       out_of_sample_date=None, data_freq=None, ret_type=None, split_method_dict=None, gp_schema=None,
                       feature_list=None, gp_operator_list=None, weighting_function=None,
                       fitness_max_stats_function=None, strategy_param_dict=None, record_num=None,
                       population_num=None, max_depth=None, min_depth=None, cxpb=None, mutpb=None, ngen=None,
                       ephemeral_constant_int_num=None, ephemeral_constant_int_min=None,
                       ephemeral_constant_int_max=None, ephemeral_constant_float_num=None,
                       ephemeral_constant_float_min=None, ephemeral_constant_float_max=None, select_mode=None,
                       select_tournsize=None, mate_mode=None, expr_mut_model=None, expr_mut_genFull_min=None,
                       expr_mut_genFull_max=None, mutate_mode=None, mutate_expr=None, mate_max=None, mutate_max=None,
                       expr_mode=None, individual_mode=None, population_mode=None):
    if strategy_name is None:
        raise NameError
    if tickers_group_name is None:
        tickers_group_name = ConstGenetic.config_dict_value_default_tickers_group_name
    if start_date is None:
        start_date = ConstGenetic.config_dict_value_default_start_date
    if end_date is None:
        end_date = ConstGenetic.config_dict_value_default_end_date
    if out_of_sample_date is None:
        out_of_sample_date = ConstGenetic.config_dict_value_default_out_of_sample_date

    if data_freq is None:
        data_freq = ConstGenetic.config_dict_value_default_frequency
    if ret_type is None:
        ret_type = ConstGenetic.config_dict_value_default_return
    if split_method_dict is None:
        split_method_dict = {
            ConstGenetic.config_dict_value_train_test_split_method_key_method: ConstGenetic.config_dict_value_train_test_split_method_value_default_method}
    if type(split_method_dict) is str:
        split_method_dict = {
            ConstGenetic.config_dict_value_train_test_split_method_key_method: split_method_dict}

    if gp_schema is None:
        gp_schema = ConstGenetic.config_dict_value_default_schema
    if feature_list is None:
        feature_list = ConstGenetic.config_dict_value_default_feature_factor

    if gp_operator_list is None:
        gp_operator_list = ConstGenetic.config_dict_value_default_gp_operator_factor

    if weighting_function is None:
        weighting_function = ConstGenetic.config_dict_value_default_weighting_function
    if fitness_max_stats_function is None:
        fitness_max_stats_function = ConstGenetic.config_dict_value_default_fitness_max_stats_function
    if strategy_param_dict is None:
        strategy_param_dict = {
            ConstGenetic.config_dict_value_strategy_param_key_num: ConstGenetic.config_dict_value_strategy_param_value_default_num,
            ConstGenetic.config_dict_value_strategy_param_order_feature: ConstGenetic.config_dict_value_strategy_param_value_default_order_feature}
    # if record_num is None:
    #     record_num = ConstGenetic.config_dict_value_default_record_num

    if population_num is None:
        population_num = ConstGenetic.config_dict_key_gp_run_param_value_default_population_num
    if max_depth is None:
        max_depth = ConstGenetic.config_dict_key_gp_run_param_value_default_expression_max_depth
    if min_depth is None:
        min_depth = ConstGenetic.config_dict_key_gp_run_param_value_default_expression_min_depth
    if cxpb is None:
        cxpb = ConstGenetic.config_dict_key_gp_run_param_value_default_cxpb
    if mutpb is None:
        mutpb = ConstGenetic.config_dict_key_gp_run_param_value_default_mutpb
    if ngen is None:
        ngen = ConstGenetic.config_dict_key_gp_run_param_value_default_ngen
    gp_run_param_dict = {
        ConstGenetic.config_dict_key_gp_run_param_key_population_num: population_num,
        ConstGenetic.config_dict_key_gp_run_param_key_expression_max_depth: max_depth,
        ConstGenetic.config_dict_key_gp_run_param_key_expression_min_depth: min_depth,
        ConstGenetic.config_dict_key_gp_run_param_key_cxpb: cxpb,
        ConstGenetic.config_dict_key_gp_run_param_key_mutpb: mutpb,
        ConstGenetic.config_dict_key_gp_run_param_key_ngen: ngen
    }

    if ephemeral_constant_int_num is None:
        ephemeral_constant_int_num = ConstGenetic.config_dict_key_gp_initial_param_value_default_ephemeral_constant_int_num
    if ephemeral_constant_int_min is None:
        ephemeral_constant_int_min = ConstGenetic.config_dict_key_gp_initial_param_value_default_ephemeral_constant_int_min
    if ephemeral_constant_int_max is None:
        ephemeral_constant_int_max = ConstGenetic.config_dict_key_gp_initial_param_value_default_ephemeral_constant_int_max

    if ephemeral_constant_float_num is None:
        ephemeral_constant_float_num = ConstGenetic.config_dict_key_gp_initial_param_value_default_ephemeral_constant_float_num
    if ephemeral_constant_float_min is None:
        ephemeral_constant_float_min = ConstGenetic.config_dict_key_gp_initial_param_value_default_ephemeral_constant_float_min
    if ephemeral_constant_float_max is None:
        ephemeral_constant_float_max = ConstGenetic.config_dict_key_gp_initial_param_value_default_ephemeral_constant_float_max
    gp_initial_param_dict = {
        ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_int_num: ephemeral_constant_int_num,
        ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_int_min: ephemeral_constant_int_min,
        ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_int_max: ephemeral_constant_int_max,
        ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_float_num: ephemeral_constant_float_num,
        ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_float_min: ephemeral_constant_float_min,
        ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_float_max: ephemeral_constant_float_max,
    }

    if select_mode is None:
        select_mode = ConstGenetic.config_dict_values_toolbox_value_default_select_mode
    if select_tournsize is None:
        select_tournsize = ConstGenetic.config_dict_values_toolbox_value_default_select_tournsize
    if mate_mode is None:
        mate_mode = ConstGenetic.config_dict_values_toolbox_value_default_mate_mode
    if expr_mut_model is None:
        expr_mut_model = ConstGenetic.config_dict_values_toolbox_value_default_expr_mut_model
    if expr_mut_genFull_min is None:
        expr_mut_genFull_min = ConstGenetic.config_dict_values_toolbox_value_default_expr_mut_genFull_min
    if expr_mut_genFull_max is None:
        expr_mut_genFull_max = ConstGenetic.config_dict_values_toolbox_value_default_expr_mut_genFull_max
    if mutate_mode is None:
        mutate_mode = ConstGenetic.config_dict_values_toolbox_value_default_mutate_mode
    if mutate_expr is None:
        mutate_expr = ConstGenetic.config_dict_values_toolbox_value_default_mutate_expr
    if mate_max is None:
        mate_max = ConstGenetic.config_dict_values_toolbox_value_default_mate_max
    if mutate_max is None:
        mutate_max = ConstGenetic.config_dict_values_toolbox_value_default_mutate_max
    if expr_mode is None:
        expr_mode = ConstGenetic.config_dict_values_toolbox_value_default_expr_mode
    if individual_mode is None:
        individual_mode = ConstGenetic.config_dict_values_toolbox_value_default_individual_mode
    if population_mode is None:
        population_mode = ConstGenetic.config_dict_values_toolbox_value_default_population_mode

    toolbox_param_dict = {
        ConstGenetic.config_dict_values_toolbox_key_select_mode: select_mode,
        ConstGenetic.config_dict_values_toolbox_key_select_tournsize: select_tournsize,
        ConstGenetic.config_dict_values_toolbox_key_mate_mode: mate_mode,
        ConstGenetic.config_dict_values_toolbox_key_expr_mut_model: expr_mut_model,
        ConstGenetic.config_dict_values_toolbox_key_expr_mut_genFull_min: expr_mut_genFull_min,
        ConstGenetic.config_dict_values_toolbox_key_expr_mut_genFull_max: expr_mut_genFull_max,
        ConstGenetic.config_dict_values_toolbox_key_mutate_mode: mutate_mode,
        ConstGenetic.config_dict_values_toolbox_key_mutate_expr: mutate_expr,
        ConstGenetic.config_dict_values_toolbox_key_mate_max: mate_max,
        ConstGenetic.config_dict_values_toolbox_key_mutate_max: mutate_max,
        ConstGenetic.config_dict_values_toolbox_key_expr_mode: expr_mode,
        ConstGenetic.config_dict_values_toolbox_key_individual_mode: individual_mode,
        ConstGenetic.config_dict_values_toolbox_key_population_mode: population_mode
    }

    # pattern mode revise
    if ConstGenetic.config_dict_value_schema_pattern_recognition in gp_schema:
        feature_list = ConstGenetic.config_dict_value_default_feature_pattern_recognition
        gp_operator_list = ConstGenetic.config_dict_value_default_gp_operator_pattern_recognition
        weighting_function = ConstGenetic.config_dict_value_default_weighting_function_pattern_recognition

    # if record_num is None:
    #     record_num = population_num
    # # record number restrict
    if record_num is None:
        if population_num < 1000:
            record_num = min(200, population_num / 5)
        elif population_num < 3000:
            record_num = 400
        else:
            record_num = 500
    config_dict = {ConstGenetic.config_dict_key_strategy_name: strategy_name,
                   ConstGenetic.config_dict_key_group_name: tickers_group_name,
                   ConstGenetic.config_dict_key_tickers: ConstFut.tickers_group_mapping[tickers_group_name],
                   ConstGenetic.config_dict_key_start_date: start_date,
                   ConstGenetic.config_dict_key_end_date: end_date,
                   ConstGenetic.config_dict_key_out_of_sample_date: out_of_sample_date,
                   ConstGenetic.config_dict_key_frequency: data_freq,
                   ConstGenetic.config_dict_key_return: ret_type,
                   ConstGenetic.config_dict_key_train_test_split_method: split_method_dict,
                   ConstGenetic.config_dict_key_schema: gp_schema,
                   ConstGenetic.config_dict_key_feature: feature_list,
                   ConstGenetic.config_dict_key_gp_operator: gp_operator_list,
                   ConstGenetic.config_dict_key_weighting_function: weighting_function,
                   ConstGenetic.config_dict_key_fitness_max_stats_function: fitness_max_stats_function,
                   ConstGenetic.config_dict_key_strategy_param: strategy_param_dict,
                   ConstGenetic.config_dict_key_gp_initial_param: gp_initial_param_dict,
                   ConstGenetic.config_dict_key_gp_run_param: gp_run_param_dict,
                   ConstGenetic.config_dict_key_toolbox: toolbox_param_dict,
                   ConstGenetic.config_dict_key_record_num: record_num}
    return config_dict


def get_config_list(strategy_name, tickers_group_name_list, data_freq_list, ret_type_list, split_method_dict_list,
                    gp_schema_list, weighting_function_list, fitness_max_stats_function_list, ngen_list, max_depth_list,
                    population_num_list):
    config_dict_list = []
    for tickers_group_name in tickers_group_name_list:
        for data_freq in data_freq_list:
            for ret_type in ret_type_list:
                for split_method_dict in split_method_dict_list:
                    for gp_schema in gp_schema_list:
                        for weighting_function in weighting_function_list:
                            for fitness_max_stats_function in fitness_max_stats_function_list:
                                for ngen in ngen_list:
                                    for max_depth in max_depth_list:
                                        for population_num in population_num_list:
                                            config_dict_list.append(
                                                config_naming_func(strategy_name=strategy_name,
                                                                   tickers_group_name=tickers_group_name,
                                                                   data_freq=data_freq,
                                                                   ret_type=ret_type,
                                                                   split_method_dict=split_method_dict,
                                                                   gp_schema=gp_schema,
                                                                   weighting_function=weighting_function,
                                                                   fitness_max_stats_function=fitness_max_stats_function,
                                                                   population_num=population_num,
                                                                   max_depth=max_depth,
                                                                   ngen=ngen,
                                                                   ))
    return config_dict_list

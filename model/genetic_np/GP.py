import pandas as pd
import numpy as np
import operator
import itertools
import datetime
import random
import time
import model
from deap import gp, creator, base, tools, algorithms

import data.LoadLocalData.load_local_data as ExtractLocalData
import model.constants.genetic as ConstGenetic
import model.constants.path as ConstPath


class DataObj:
    """
    a data object to save the feature
    """

    def __init__(self, data_dict={}):
        if len(data_dict) == 0:
            data_dict = {'open': pd.DataFrame(), 'high': pd.DataFrame(), 'low': pd.DataFrame(), 'close': pd.DataFrame(),
                         'volume': pd.DataFrame(), 'amount': pd.DataFrame(), 'I': pd.DataFrame()}
        for k, v in data_dict.items():
            setattr(self, k, v)


class GeneticProgrammingData:
    """
    a comprehensive data object derived from the DataObj
    main job:
        1. load data feature
        2. load asset return matrix
        3. split data into train and test set
        4. add I (matrix with all value 1) if schema is pattern recognition
    """

    def __init__(self, config_dict):
        self.train_data = DataObj()
        self.test_data = DataObj()
        self.all_data = DataObj()
        self.out_of_sample_data = DataObj()
        self.total_data = DataObj()
        self.config = config_dict
        #  out of sample data load in
        data_dict = self.__load_out_of_sample_feature()
        for k, v in data_dict.items():
            setattr(self.out_of_sample_data, k, v)
        # total data load in
        data_dict = self.__load_total_feature()
        for k, v in data_dict.items():
            setattr(self.total_data, k, v)
        # in sample train data and test data
        data_dict = self.__load_in_sample_feature()
        for k, v in data_dict.items():
            setattr(self.all_data, k, v)

        self.__split_data()
        self.__remove_index()
        if ConstGenetic.config_dict_value_schema_pattern_recognition in self.config[
            ConstGenetic.config_dict_key_schema]:
            setattr(self.train_data, 'I',
                    pd.DataFrame(1, index=self.train_data.ret.index, columns=self.train_data.ret.columns))
            setattr(self.test_data, 'I',
                    pd.DataFrame(1, index=self.test_data.ret.index, columns=self.test_data.ret.columns))
            setattr(self.all_data, 'I',
                    pd.DataFrame(1, index=self.all_data.ret.index, columns=self.all_data.ret.columns))
            setattr(self.out_of_sample_data, 'I',
                    pd.DataFrame(1, index=self.out_of_sample_data.ret.index,
                                 columns=self.out_of_sample_data.ret.columns))
            setattr(self.total_data, 'I',
                    pd.DataFrame(1, index=self.total_data.ret.index, columns=self.total_data.ret.columns))

    def __load_in_sample_feature(self):
        data_dict = {}
        freq = self.config[ConstGenetic.config_dict_key_frequency]

        ret_str = self.config[ConstGenetic.config_dict_key_return]
        file_location = ConstPath.input_data_return_pct_path + 'freq_' + freq + '_' + ret_str + '_ret.csv'
        data = ExtractLocalData.load_local_factor_csv(file_location=file_location, input_as_matrix=True,
                                                      output_as_matrix=True)
        data = data[[x for x in data.columns if x in self.config[ConstGenetic.config_dict_key_tickers]]]
        data = data.dropna(how='all')
        try:
            data = data[data.index >= datetime.date(int(self.config[ConstGenetic.config_dict_key_start_date][:4]),
                                                    int(self.config[ConstGenetic.config_dict_key_start_date][5:7]),
                                                    int(self.config[ConstGenetic.config_dict_key_start_date][-2:]))]
            data = data[data.index <= datetime.date(int(self.config[ConstGenetic.config_dict_key_end_date][:4]),
                                                    int(self.config[ConstGenetic.config_dict_key_end_date][5:7]),
                                                    int(self.config[ConstGenetic.config_dict_key_end_date][-2:]))]
        except:
            data = data[data.index >= datetime.datetime.strptime(
                self.config[ConstGenetic.config_dict_key_start_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
            data = data[data.index <= datetime.datetime.strptime(
                self.config[ConstGenetic.config_dict_key_end_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
        data_dict[ConstGenetic.config_dict_key_return] = data
        df_model=pd.DataFrame(data.copy().reset_index().iloc[:,0])
        for feature in self.config[ConstGenetic.config_dict_key_feature]:
            file_location = ConstPath.input_data_feature_path + 'freq_' + freq + '_' + feature + '.csv'
            data = ExtractLocalData.load_local_factor_csv(file_location=file_location,
                                                          input_as_matrix=True, output_as_matrix=True)
            data = data[[x for x in data.columns if x in self.config[ConstGenetic.config_dict_key_tickers]]]
            data = data.dropna(how='all')
            try:
                data = data[data.index >= datetime.date(int(self.config[ConstGenetic.config_dict_key_start_date][:4]),
                                                        int(self.config[ConstGenetic.config_dict_key_start_date][5:7]),
                                                        int(self.config[ConstGenetic.config_dict_key_start_date][-2:]))]
                data = data[data.index <= datetime.date(int(self.config[ConstGenetic.config_dict_key_end_date][:4]),
                                                        int(self.config[ConstGenetic.config_dict_key_end_date][5:7]),
                                                        int(self.config[ConstGenetic.config_dict_key_end_date][-2:]))]
            except:
                data = data[data.index >= datetime.datetime.strptime(
                    self.config[ConstGenetic.config_dict_key_start_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
                data = data[data.index <= datetime.datetime.strptime(
                    self.config[ConstGenetic.config_dict_key_end_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
            data=pd.merge(left=df_model,right=data.reset_index(),how='left',on=['Trade_DT'])
            data.index=data['Trade_DT']
            data=data.iloc[:,1:]
            data_dict[feature] = data
        return data_dict

    def __load_out_of_sample_feature(self):
        data_dict = {}
        freq = self.config[ConstGenetic.config_dict_key_frequency]

        ret_str = self.config[ConstGenetic.config_dict_key_return]
        file_location = ConstPath.input_data_return_pct_path + 'freq_' + freq + '_' + ret_str + '_ret.csv'
        data = ExtractLocalData.load_local_factor_csv(file_location=file_location, input_as_matrix=True,
                                                      output_as_matrix=True)
        data = data[[x for x in data.columns if x in self.config[ConstGenetic.config_dict_key_tickers]]]
        data = data.dropna(how='all')
        try:
            data = data[data.index >= datetime.date(int(self.config[ConstGenetic.config_dict_key_end_date][:4]),
                                                    int(self.config[ConstGenetic.config_dict_key_end_date][5:7]),
                                                    int(self.config[ConstGenetic.config_dict_key_end_date][-2:]))]
            data = data[
                data.index <= datetime.date(int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][:4]),
                                            int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][5:7]),
                                            int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][-2:]))]
        except:
            data = data[data.index >= datetime.datetime.strptime(
                self.config[ConstGenetic.config_dict_key_end_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
            data = data[data.index <= datetime.datetime.strptime(
                self.config[ConstGenetic.config_dict_key_out_of_sample_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]

        data_dict[ConstGenetic.config_dict_key_return] = data
        df_model = pd.DataFrame(data.copy().reset_index().iloc[:, 0])
        for feature in self.config[ConstGenetic.config_dict_key_feature]:
            file_location = ConstPath.input_data_feature_path + 'freq_' + freq + '_' + feature + '.csv'
            data = ExtractLocalData.load_local_factor_csv(file_location=file_location,
                                                          input_as_matrix=True, output_as_matrix=True)
            data = data[[x for x in data.columns if x in self.config[ConstGenetic.config_dict_key_tickers]]]
            data = data.dropna(how='all')
            try:
                data = data[data.index >= datetime.date(int(self.config[ConstGenetic.config_dict_key_end_date][:4]),
                                                        int(self.config[ConstGenetic.config_dict_key_end_date][5:7]),
                                                        int(self.config[ConstGenetic.config_dict_key_end_date][-2:]))]
                data = data[
                    data.index <= datetime.date(int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][:4]),
                                                int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][5:7]),
                                                int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][-2:]))]
            except:
                data = data[data.index >= datetime.datetime.strptime(
                    self.config[ConstGenetic.config_dict_key_end_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
                data = data[data.index <= datetime.datetime.strptime(
                    self.config[ConstGenetic.config_dict_key_out_of_sample_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
            data = pd.merge(left=df_model, right=data.reset_index(), how='left', on=['Trade_DT'])
            data.index = data['Trade_DT']
            data = data.iloc[:, 1:]
            data_dict[feature] = data
        return data_dict

    def __load_total_feature(self):
        data_dict = {}
        freq = self.config[ConstGenetic.config_dict_key_frequency]
        ret_str = self.config[ConstGenetic.config_dict_key_return]
        file_location = ConstPath.input_data_return_pct_path + 'freq_' + freq + '_' + ret_str + '_ret.csv'
        data = ExtractLocalData.load_local_factor_csv(file_location=file_location, input_as_matrix=True,
                                                      output_as_matrix=True)
        data = data[[x for x in data.columns if x in self.config[ConstGenetic.config_dict_key_tickers]]]
        data = data.dropna(how='all')
        try:
            data = data[data.index >= datetime.date(int(self.config[ConstGenetic.config_dict_key_start_date][:4]),
                                                    int(self.config[ConstGenetic.config_dict_key_start_date][5:7]),
                                                    int(self.config[ConstGenetic.config_dict_key_start_date][-2:]))]
            data = data[
                data.index <= datetime.date(int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][:4]),
                                            int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][5:7]),
                                            int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][-2:]))]
        except:
            data = data[data.index >= datetime.datetime.strptime(
                self.config[ConstGenetic.config_dict_key_start_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
            data = data[data.index <= datetime.datetime.strptime(
                self.config[ConstGenetic.config_dict_key_out_of_sample_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]

        data_dict[ConstGenetic.config_dict_key_return] = data
        df_model = pd.DataFrame(data.copy().reset_index().iloc[:, 0])

        for feature in self.config[ConstGenetic.config_dict_key_feature]:
            file_location = ConstPath.input_data_feature_path + 'freq_' + freq + '_' + feature + '.csv'
            data = ExtractLocalData.load_local_factor_csv(file_location=file_location,
                                                          input_as_matrix=True, output_as_matrix=True)
            data = data[[x for x in data.columns if x in self.config[ConstGenetic.config_dict_key_tickers]]]
            data = data.dropna(how='all')
            try:
                data = data[data.index >= datetime.date(int(self.config[ConstGenetic.config_dict_key_start_date][:4]),
                                                        int(self.config[ConstGenetic.config_dict_key_start_date][5:7]),
                                                        int(self.config[ConstGenetic.config_dict_key_start_date][-2:]))]
                data = data[
                    data.index <= datetime.date(int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][:4]),
                                                int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][5:7]),
                                                int(self.config[ConstGenetic.config_dict_key_out_of_sample_date][-2:]))]
            except:
                data = data[data.index >= datetime.datetime.strptime(
                    self.config[ConstGenetic.config_dict_key_start_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
                data = data[data.index <= datetime.datetime.strptime(
                    self.config[ConstGenetic.config_dict_key_out_of_sample_date] + " 00:00:00", "%Y-%m-%d %H:%M:%S")]
            data = pd.merge(left=df_model, right=data.reset_index(), how='left', on=['Trade_DT'])
            data.index = data['Trade_DT']
            data = data.iloc[:, 1:]
            data_dict[feature] = data
        return data_dict

    def __split_data(self):
        split_param_list = [str(k) + '=' + str(v) for k, v in
                            self.config[ConstGenetic.config_dict_key_train_test_split_method].items()]
        param_str = ','.join(split_param_list[1:])
        method = split_param_list[0].split('=')[1]
        exec(f"from model.genetic.function.data_split_function import {method}")

        for feature in self.config[ConstGenetic.config_dict_key_feature] + [ConstGenetic.config_dict_key_return]:
            data = getattr(self.all_data, feature)
            if len(param_str) == 0:
                data_train, data_test = eval(f"{method}(data=data)")
            else:
                data_train, data_test = eval(f"{method}(data=data,{param_str})")
            setattr(self.train_data, feature, data_train)
            setattr(self.test_data, feature, data_test)

    def __remove_index(self):
        for feature in self.config[ConstGenetic.config_dict_key_feature] + [ConstGenetic.config_dict_key_return]:
            data = getattr(self.all_data, feature)
            if 'time_index' not in dir(self.all_data):
                setattr(self.all_data, 'time_index', data.index)
            setattr(self.all_data, feature, data.reset_index(drop=True))

            data = getattr(self.train_data, feature)
            if 'time_index' not in dir(self.train_data):
                setattr(self.train_data, 'time_index', data.index)
            setattr(self.train_data, feature, data.reset_index(drop=True))

            data = getattr(self.test_data, feature)
            if 'time_index' not in dir(self.test_data):
                setattr(self.test_data, 'time_index', data.index)
            setattr(self.test_data, feature, data.reset_index(drop=True))

            data = getattr(self.out_of_sample_data, feature)
            if 'time_index' not in dir(self.out_of_sample_data):
                setattr(self.out_of_sample_data, 'time_index', data.index)
            setattr(self.out_of_sample_data, feature, data.reset_index(drop=True))

            data = getattr(self.total_data, feature)
            if 'time_index' not in dir(self.total_data):
                setattr(self.total_data, 'time_index', data.index)
            setattr(self.total_data, feature, data.reset_index(drop=True))


class GeneticProgramming(GeneticProgrammingData):
    """
    Genetic Programming Plan Object
    initial:
        1. load feature
        2. load pct
        3. split the data into train data and test data
        4. initial pset: add terminals (input data) and operators (calculation methods)
        5. initial toolbox: set the gp parameters  (formula number, depth and...)
        6. import fitness function (from model.genetic.functions.fitness_function.py)
        6. gp_main: the running function of genetic programming plan
    """

    def __init__(self, config_dict):
        # load feature
        super().__init__(config_dict)
        # initial pset and toolbox
        self.pset = self.init_primitive_set()
        self.toolbox = self.init_toolbox()

    def evalfunc(self, individual):
        # complie the formula to a function object

        function = self.toolbox.compile(expr=individual)
        # run the formula result
        if ConstGenetic.config_dict_value_schema_pattern_recognition in self.config[
            ConstGenetic.config_dict_key_schema]:
            # res = eval(f"function(I=self.train_data.I)")
            data_obj_func_list = []
            non_data_obj_func_list = []
            for func in self.config[ConstGenetic.config_dict_key_gp_operator]:
                # import funcition from the function py module
                exec(f"from model.genetic.function.gp_operator import {func}")
                if 'data_obj' in eval(func).__annotations__.keys():
                    data_obj_func_list.append(func)
                else:
                    non_data_obj_func_list.append(func)
            func_str = str(individual).replace(")", ",data_obj=data_obj_we_need_to_change)")
            func_str = func_str.replace("(,", "(")
            for str1 in non_data_obj_func_list:
                func_str = func_str.replace(str1 + "(data_obj=data_obj_we_need_to_change)", str1 + "()")

            ###########################################################################
            # all data
            I = self.train_data.I
            res = eval(func_str.replace("data_obj_we_need_to_change", "self.train_data"))
            if "short" in self.config[ConstGenetic.config_dict_key_schema]:
                res = res.mul(-1)
        else:
            function = self.toolbox.compile(expr=individual)
            res = eval(
                f"function({','.join([i + '=' + 'self.train_data.' + i for i in self.config[ConstGenetic.config_dict_key_feature]])})")

        # import the fitness function and calculate the fitness statistics
        from model.genetic.function.fitness_function import fitness_function_structure
        result = fitness_function_structure(factor_value=res,
                                            pct=getattr(self.train_data, ConstGenetic.config_dict_key_return),
                                            data_obj=self.train_data, config_dict=self.config)

        # exec(
        #     f"from model.genetic.function.fitness_function import {self.config[ConstGenetic.config_dict_key_fitness_function]}")
        # fitness_func = eval(self.config[ConstGenetic.config_dict_key_fitness_function])
        # result = fitness_func(res=res, pct=getattr(self.train_data, ConstGenetic.config_dict_key_return),
        #                       data_obj=self.train_data)
        return result,

    def init_primitive_set(self):

        # initial gplearn object input and output type and input rename
        if ConstGenetic.config_dict_value_schema_pattern_recognition in self.config[
            ConstGenetic.config_dict_key_schema]:
            # if pattern
            pset = gp.PrimitiveSetTyped("MAIN", [pd.DataFrame], pd.DataFrame)
            pset.renameArguments(ARG0='I')
        else:
            # if not pattern
            pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(pd.DataFrame, len(
                self.config[ConstGenetic.config_dict_key_feature])), pd.DataFrame)
            for i in range(len(self.config[ConstGenetic.config_dict_key_feature])):
                exec(f"pset.renameArguments(ARG{i}='{self.config[ConstGenetic.config_dict_key_feature][i]}')")

        # load gp operator, domain and range
        for func in self.config[ConstGenetic.config_dict_key_gp_operator]:
            # import funcition from the function py module
            exec(f"from model.genetic.function.gp_operator import {func}")
            try:
                input_list = [type(x) for x in eval(func).__defaults__ if x is not None]
            except TypeError:
                input_list = []
            try:
                pset.addPrimitive(eval(func), input_list, type(eval(func + '()')))
            except Exception as e:
                pset.addPrimitive(eval(func), input_list, pd.DataFrame)

        # addEphemeralConstant Int
        l1 = self.config[ConstGenetic.config_dict_key_gp_initial_param][
            ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_int_num]
        if l1 > 0:
            try:
                for i in range(l1):
                    pset.addEphemeralConstant("int_num" + str(i),
                                              lambda: random.randint(
                                                  self.config[ConstGenetic.config_dict_key_gp_initial_param][
                                                      ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_int_min],
                                                  self.config[ConstGenetic.config_dict_key_gp_initial_param][
                                                      ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_int_max]),
                                              int)
            except:
                pass

        # addEphemeralConstant Float
        l2 = self.config[ConstGenetic.config_dict_key_gp_initial_param][
            ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_float_num]

        if l2 > 0:
            try:
                for i in range(l2):
                    pset.addEphemeralConstant("float_num" + str(i),
                                              lambda: random.uniform(
                                                  self.config[ConstGenetic.config_dict_key_gp_initial_param][
                                                      ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_float_min],
                                                  self.config[ConstGenetic.config_dict_key_gp_initial_param][
                                                      ConstGenetic.config_dict_key_gp_initial_param_key_ephemeral_constant_float_max]),
                                              float)
            except:
                pass

        return pset

    def init_toolbox(self):

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        eval(f"toolbox.register(\"evaluate\", self.evalfunc)")

        toolbox.register("select", eval(
            self.config[ConstGenetic.config_dict_key_toolbox][ConstGenetic.config_dict_values_toolbox_key_select_mode]),
                         tournsize=self.config[ConstGenetic.config_dict_key_toolbox][
                             ConstGenetic.config_dict_values_toolbox_key_select_tournsize])  # select数量
        toolbox.register("mate", eval(
            self.config[ConstGenetic.config_dict_key_toolbox][ConstGenetic.config_dict_values_toolbox_key_mate_mode]))
        toolbox.register("expr_mut", eval(self.config[ConstGenetic.config_dict_key_toolbox][
                                              ConstGenetic.config_dict_values_toolbox_key_expr_mut_model]),
                         min_=self.config[ConstGenetic.config_dict_key_toolbox][
                             ConstGenetic.config_dict_values_toolbox_key_expr_mut_genFull_min],
                         max_=self.config[ConstGenetic.config_dict_key_toolbox][
                             ConstGenetic.config_dict_values_toolbox_key_expr_mut_genFull_max])
        toolbox.register("mutate", eval(
            self.config[ConstGenetic.config_dict_key_toolbox][ConstGenetic.config_dict_values_toolbox_key_mutate_mode]),
                         expr=eval(self.config[ConstGenetic.config_dict_key_toolbox][
                                       ConstGenetic.config_dict_values_toolbox_key_mutate_expr]), pset=self.pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                                max_value=self.config[ConstGenetic.config_dict_key_toolbox][
                                                    ConstGenetic.config_dict_values_toolbox_key_mate_max]))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                                  max_value=self.config[ConstGenetic.config_dict_key_toolbox][
                                                      ConstGenetic.config_dict_values_toolbox_key_mutate_max]))

        toolbox.register("expr", eval(
            self.config[ConstGenetic.config_dict_key_toolbox][ConstGenetic.config_dict_values_toolbox_key_expr_mode]),
                         pset=self.pset,
                         min_=self.config[ConstGenetic.config_dict_key_gp_run_param][
                             ConstGenetic.config_dict_key_gp_run_param_key_expression_min_depth],
                         max_=self.config[ConstGenetic.config_dict_key_gp_run_param][
                             ConstGenetic.config_dict_key_gp_run_param_key_expression_max_depth])
        toolbox.register("individual", eval(self.config[ConstGenetic.config_dict_key_toolbox][
                                                ConstGenetic.config_dict_values_toolbox_key_individual_mode]),
                         creator.Individual, toolbox.expr)
        toolbox.register("population", eval(self.config[ConstGenetic.config_dict_key_toolbox][
                                                ConstGenetic.config_dict_values_toolbox_key_population_mode]), list,
                         toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        return toolbox

    def gp_main(self):
        random.seed(int(time.time()))
        # randomly generate the formula
        pop = self.toolbox.population(n=self.config[ConstGenetic.config_dict_key_gp_run_param][
            ConstGenetic.config_dict_key_gp_run_param_key_population_num])
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.nanmean)
        mstats.register("std", np.nanstd)
        mstats.register("min", np.nanmin)
        mstats.register("max", np.nanmax)

        # load the mutpb,cxpb,ngen parameter and run the gp plan
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.config[ConstGenetic.config_dict_key_gp_run_param][
            ConstGenetic.config_dict_key_gp_run_param_key_cxpb],
                                       mutpb=self.config[ConstGenetic.config_dict_key_gp_run_param][
                                           ConstGenetic.config_dict_key_gp_run_param_key_mutpb],
                                       ngen=self.config[ConstGenetic.config_dict_key_gp_run_param][
                                           ConstGenetic.config_dict_key_gp_run_param_key_ngen],
                                       stats=mstats, verbose=True)
        return pop, log

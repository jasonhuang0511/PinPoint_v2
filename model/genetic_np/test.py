import sys

sys.path.append('..')
sys.path.append('C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2')

import model.genetic.GP as GP
import data.ConstantData.future_basic_information as ConstFutBasic
import model.constants.genetic as ConstGenetic
import datetime
from deap import tools

config_dict_pattern = {
    'tickers': ConstFutBasic.fut_code_list,
    'start_date': '2018-01-01',
    'end_date': '2022-01-01',
    'freq': '1D',
    'ret': 'close',
    'split_method': {'method': 'ts_split', 'ratio': 0.8},
    'feature': ['open', 'close', 'high', 'low', 'volume'],
    'gp_schema': 'pattern_recognition',
    'gp_operator': [
        'both_test', 'either_test', 'both3_test', 'both4_test', 'both5_test', 'ti_close_over_mean_test',
        'ti_close_below_ma_test', 'ti_ma1_over_ma2_test', 'int_1',
        'int_10', 'int_1000', 'int_11', 'int_12', 'int_120', 'int_13', 'int_14', 'int_15',
        'int_16', 'int_2', 'int_20', 'int_252', 'int_3', 'int_30', 'int_4', 'int_40', 'int_5', 'int_504', 'int_6',
        'int_60', 'int_7', 'int_8', 'int_9',
        'int_90', 'float_001', 'float_002', 'float_003', 'float_004', 'float_005', 'float_006', 'float_007',
        'float_008',
        'float_009', 'float_01', 'float_015', 'float_02', 'float_025', 'float_03', 'float_035', 'float_04',
        'float_045', 'float_05', 'float_06', 'float_07', 'float_08', 'float_09'],
    'FitnessFunction': 'max_ir_ret',
    'ResultSaveFunction': 'save_ts_pattern_recognition_nv',
    'gp_initial_param': {'EphemeralConstantIntNum': 15, 'EphemeralConstantIntMin': 1,
                         'EphemeralConstantIntMax': 25, 'EphemeralConstantFloatNum': 10,
                         'EphemeralConstantFloatMin': 0,
                         'EphemeralConstantFloatMax': 1},
    'gp_run_param': {'population_num': 50, 'max_depth': 6, 'min_depth': 2, 'cxpb': 0.5, 'mutpb': 0.1, 'ngen': 2},
    'record_num': 100
}
gp_obj = GP.GeneticProgramming(config_dict=config_dict_pattern)

if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S'))
    print("start_programming")
    pop, log = gp_obj.gp_main()

    top = tools.selBest(pop, k=gp_obj.config[ConstGenetic.config_dict_key_record_num])
    noDupes = []
    [noDupes.append(i) for i in top if not noDupes.count(i)]
    exec(
        f"from model.genetic.function.result_save_function import {gp_obj.config[ConstGenetic.config_dict_key_result_save_function]}")
    eval(f"{gp_obj.config[ConstGenetic.config_dict_key_result_save_function]}(result=noDupes,gp_obj=gp_obj)")

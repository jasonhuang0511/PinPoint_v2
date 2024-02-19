import sys

sys.path.append('..')
sys.path.append('C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2')

import model.genetic.GP as GP
import model.constants.genetic as ConstGenetic
import model.genetic.config_list as Config_List
import datetime
from deap import tools
import time
import random
import click


@click.command()
@click.option("--strategy_name", "-sn", default='test', required=False, help="strategy name (self-defined)")
@click.option("--tickers_group_name", "-t", default=None, required=False, help="backtest tickers group name")
@click.option("--split_method", "-sm", default=None, required=False,
              help="method of splitting data into train and test")
@click.option("--schema", "-sc", default=None, required=False, help="strategy type")
@click.option("-weighting_func", "-wf", default=None, required=False, help="weighting function of factor")
@click.option("--indicator", "-ind", default=None, required=False, help="fitness_indicator which needs to max")
@click.option("--ngen", "-ngen", default=None, required=False, help="generation of gp")
@click.option("--population_num", "-pop", default=None, required=False, help="initial population number")
@click.option("--max_depth", "-dep", default=None, required=False, help="formula tree depth")
def run_gp(strategy_name, tickers_group_name, split_method, schema, weighting_func, indicator, ngen, population_num,
           max_depth):
    config_dict = Config_List.config_naming_func(strategy_name=strategy_name, tickers_group_name=tickers_group_name,
                                                 split_method_dict=split_method, gp_schema=schema,
                                                 weighting_function=weighting_func,
                                                 fitness_max_stats_function=indicator,
                                                 population_num=int(population_num),
                                                 max_depth=int(max_depth), ngen=int(ngen))
    # random start time
    time.sleep(random.random() * 10)
    # initial the genetic programming object, load data, initial setting of toolbox and pset
    gp_obj = GP.GeneticProgramming(config_dict=config_dict)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("start_programming")
    # run the genetic programming
    pop, log = gp_obj.gp_main()
    # select the top k best performance formula
    top = tools.selBest(pop, k=int(gp_obj.config[ConstGenetic.config_dict_key_record_num]))

    # drop the duplicate result
    noDupes = []
    [noDupes.append(i) for i in top if not noDupes.count(i)]

    # save the formula result
    from model.genetic.function.result_save_function import result_save_function

    result_save_function(gp_obj, noDupes)
    # eval(f"{gp_obj.config[ConstGenetic.config_dict_key_result_save_function]}(result=noDupes,gp_obj=gp_obj)")


if __name__ == '__main__':
    run_gp()

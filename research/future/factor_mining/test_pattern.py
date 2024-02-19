from multiprocessing.dummy import Pool
import pandas as pd
import numpy as np
import operator
import itertools
import datetime
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import random
import time
import math
import warnings

import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFutBasic

warnings.filterwarnings('ignore')

tickers_list = ConstFutBasic.fut_code_list
start_date = '2018-01-01'
end_date = '2021-09-16'
fee = 0.0005
out_of_sample_size = 0
freq = 'D'
# basic price volume factor


close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq=freq, index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close = close.iloc[:-1, ]
close_all = close.copy()
close = close_all.iloc[-1 * out_of_sample_size:, ]

index_all = close_all.index
index_test = close.index

close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq=freq, index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close = close.iloc[:-1, ]
close.index = range(len(close))
close_all = close.copy()
close = close_all.iloc[-1 * out_of_sample_size:, ]

open = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq=freq, index=1, ret_index=False)
open['Fut_code'] = open['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open = open.pivot_table(index='Trade_DT', columns='Fut_code', values=open.columns[2])
open = open.iloc[:-1, ]
open.index = range(len(open))
open_all = open.copy()
open = open_all.iloc[-1 * out_of_sample_size:, ]

high = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='high', freq=freq, index=1, ret_index=False)
high['Fut_code'] = high['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
high = high.pivot_table(index='Trade_DT', columns='Fut_code', values=high.columns[2])
high = high.iloc[:-1, ]
high.index = range(len(high))
high_all = high.copy()
high = high_all.iloc[-1 * out_of_sample_size:, ]

low = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='low',
                                        freq='D', index=1, ret_index=False)
low['Fut_code'] = low['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
low = low.pivot_table(index='Trade_DT', columns='Fut_code', values=low.columns[2])
low = low.iloc[:-1, ]
low.index = range(len(low))
low_all = low.copy()
low = low_all.iloc[-1 * out_of_sample_size:, ]

amount = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='amount', freq=freq, index=1, ret_index=False)
amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])
amount = amount.iloc[:-1, ].reset_index(drop=True)
amount_all = amount.copy()
amount = amount_all.iloc[-1 * out_of_sample_size:, ]

open_pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                             key_word='open', freq=freq, index=1, ret_index=False)
open_pct['Fut_code'] = open_pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open_pct = open_pct.pivot_table(index='Trade_DT', columns='Fut_code', values=open_pct.columns[2])
close_pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                              key_word='close', freq=freq, index=1, ret_index=False)
close_pct['Fut_code'] = close_pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close_pct = close_pct.pivot_table(index='Trade_DT', columns='Fut_code', values=close_pct.columns[2])
ret = close_pct * (1 - fee) / open_pct / (1 + fee) - 1
ret = ret.iloc[1:, ].reset_index(drop=True)

ret_all = ret.copy()
ret = ret_all.iloc[-1 * out_of_sample_size:, ]

I = pd.DataFrame(1, index=ret.index, columns=ret.columns)


class DataPriceVolume:
    def __init__(self, open, close, high, low, amount, ret):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.amount = amount
        self.ret = ret
        self.I=pd.DataFrame(1,index=self.ret.index,columns=self.ret.columns)


d = DataPriceVolume(open, close, high, low, amount, ret)




pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(DataPriceVolume, 1), pd.DataFrame)


# 同时两个信号
def both(d,signal1, signal2):
    return d.I.where(signal1.add(signal2) >= 2, 0)


pset.addPrimitive(both, [DataPriceVolume,pd.DataFrame, pd.DataFrame], pd.DataFrame)


def either(d,signal1, signal2):
    return d.I.where(signal1.add(signal2) >= 1, 0)


pset.addPrimitive(either, [DataPriceVolume,pd.DataFrame, pd.DataFrame], pd.DataFrame)


# 收于均线上方
def closeoverma(d,N):
    ma = d.close.rolling(N).mean()
    sig = d.I.where(d.close > ma, 0)
    return sig


pset.addPrimitive(closeoverma, [DataPriceVolume,int], pd.DataFrame)


# 收于均线下方
def closebelowma(d,N):
    ma = d.close.rolling(N).mean()
    sig = d.I.where(d.close < ma, 0)
    return sig


pset.addPrimitive(closebelowma, [DataPriceVolume,int], pd.DataFrame)


# 均线上穿
def ma1overma2(d,N1, N2):
    ma1 = d.close.rolling(N1).mean()
    ma2 = d.close.rolling(N2).mean()
    sig = d.I.where(ma1 > ma2, 0)
    return sig


pset.addPrimitive(ma1overma2, [DataPriceVolume,int, int], pd.DataFrame)


# 长上影线
def longupshadow(d,pct):
    maxop = d.open.where(d.open > d.close, d.close)
    upshadow = d.high.sub(maxop)
    sig = d.I.where(upshadow.div(d.high.sub(low)) > pct, 0)
    return sig


pset.addPrimitive(longupshadow, [DataPriceVolume,float], pd.DataFrame)




# 整数
def int_1():
    return 1


pset.addPrimitive(int_1, [], int)


def int_2():
    return 2


pset.addPrimitive(int_2, [], int)


def int_3():
    return 3


pset.addPrimitive(int_3, [], int)


def int_4():
    return 4


pset.addPrimitive(int_4, [], int)



def int_20():
    return 20


pset.addPrimitive(int_20, [], int)


def int_21():
    return 21


pset.addPrimitive(int_21, [], int)


# 小数
def float_005():
    return 0.05


pset.addPrimitive(float_005, [], float)


def float_01():
    return 0.1


pset.addPrimitive(float_01, [], float)


def float_015():
    return 0.15


pset.addPrimitive(float_015, [], float)


def float_02():
    return 0.2


pset.addPrimitive(float_02, [], float)


def float_05():
    return 0.5


pset.addPrimitive(float_05, [], float)


def float_06():
    return 0.6


pset.addPrimitive(float_06, [], float)


def float_07():
    return 0.7


pset.addPrimitive(float_07, [], float)


def float_08():
    return 0.8


pset.addPrimitive(float_08, [], float)


def float_09():
    return 0.9


pset.addPrimitive(float_09, [], float)


def floatsub1(pct):
    return 1 - pct

pset.addPrimitive(floatsub1, [float], float)


def datacreate():
    return DataPriceVolume(open, close, high, low, amount, ret)

pset.addPrimitive(datacreate, [], DataPriceVolume)

pset.addTerminal(d,DataPriceVolume,'name')


pset.addEphemeralConstant("const2", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("float2", lambda: random.uniform(0, 1), float)

pset.renameArguments(ARG0='I')



creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=10)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, d):
    function = toolbox.compile(expr=individual)
    res = function(d)
    res1 = res.where(d.ret.isnull() == False).dropna(axis=0, how='all')
    amountrank = (-1 * d.amount).where(res1 > 0).rank(axis=1)

    ret_cond = ret.where(amountrank <= 10)
    ret_mean = ret_cond.sum(axis=1) / 10
    if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
        return -np.inf, -np.inf,
    else:
        return ret_mean.dropna().mean() / ret_mean.dropna().std(), ret_mean.fillna(0).sum(),


toolbox.register("evaluate", evalSymbReg, d=d)
# toolbox.register("select1", tools.selBest)#select数量
toolbox.register("select", tools.selTournament, tournsize=4)  # select数量
##两棵树每棵树各选一颗，然后交换其中一个
toolbox.register("mate", gp.cxOnePoint)
##产生一个表达式，所有的叶子结点有相同的长度
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
##不能交配过长
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))

pset.addTerminal(idem, [MyClass], MyClass)

def main():
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats用来汇报结果
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.nanmean)
    mstats.register("std", np.nanstd)
    mstats.register("min", np.nanmin)
    mstats.register("max", np.nanmax)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=1, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(log)
    return pop, log, hof


if __name__ != "__main__":
    s = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
    print(s)
    random.seed(int(time.time()))
    pool = Pool(10)
    toolbox.register("map", pool.map)
    print("start_programming")
    pop, log, hof = main()

    top = tools.selBest(pop, k=200)
    noDupes = []
    [noDupes.append(i) for i in top if not noDupes.count(i)]
    result_all = pd.DataFrame()
    save_location = 'C:\\Users\\jason.huang\\research\\data_mining\\pattern_technical_indicator\\factor_' + s + '.csv'

    for i in range(len(noDupes)):
        tree = gp.PrimitiveTree(noDupes[i])
        print(str(tree))
        function = gp.compile(tree, pset)

        res = function(I)
        res1 = res.where(ret_all.isnull() == False).dropna(axis=0, how='all')
        amountrank = (-1 * amount).where(res1 > 0).rank(axis=1)
        ret_cond = ret_all.where(amountrank <= 10)
        ret_mean = ret_cond.sum(axis=1) / 10
        cum_pnl = ret_mean.cumsum() + 1
        cum_pnl = pd.DataFrame(cum_pnl)
        cum_pnl.index = index_test
        cum_pnl.columns = [str(tree)]
        result_all = pd.concat([result_all, cum_pnl], axis=1)
        result_all.to_csv(save_location)

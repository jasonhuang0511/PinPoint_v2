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
import math
import warnings

import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFutBasic

warnings.filterwarnings('ignore')

tickers_list = ConstFutBasic.fut_code_list
start_date = '2018-01-01'
end_date = '2022-08-31'

# predict percent
pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                        key_word='close', freq='D', index=1, ret_index=True)
pct['Fut_code'] = pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct = pct.pivot_table(index='Trade_DT', columns='Fut_code', values=pct.columns[2])
pct = pct.iloc[1:, :]
pct.index = range(len(pct))

# basic price volume factor

close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq='D', index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close = close.iloc[:-1, ]
close.index = range(len(close))

open = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq='D', index=1, ret_index=False)
open['Fut_code'] = open['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open = open.pivot_table(index='Trade_DT', columns='Fut_code', values=open.columns[2])
open = open.iloc[:-1, ]
open.index = range(len(open))

high = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='high', freq='D', index=1, ret_index=False)
high['Fut_code'] = high['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
high = high.pivot_table(index='Trade_DT', columns='Fut_code', values=high.columns[2])
high = high.iloc[:-1, ]
high.index = range(len(high))

low = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='low',
                                        freq='D', index=1, ret_index=False)
low['Fut_code'] = low['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
low = low.pivot_table(index='Trade_DT', columns='Fut_code', values=low.columns[2])
low = low.iloc[:-1, ]
low.index = range(len(low))

settle = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='settle',
                                           freq='D', index=1, ret_index=False)
settle['Fut_code'] = settle['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
settle = settle.pivot_table(index='Trade_DT', columns='Fut_code', values=settle.columns[2])
settle = settle.iloc[:-1, ]
settle.index = range(len(settle))

volume = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='vol', freq='D', index=1, ret_index=False)
volume['Fut_code'] = volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume = volume.pivot_table(index='Trade_DT', columns='Fut_code', values=volume.columns[2])
volume = volume.iloc[:-1, ]
volume.index = range(len(volume))

oi = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                       key_word='openinterest', freq='D', index=1, ret_index=False)
oi['Fut_code'] = oi['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
oi = oi.pivot_table(index='Trade_DT', columns='Fut_code', values=oi.columns[2])
oi = oi.iloc[:-1, ].reset_index(drop=True)

vwap_amount = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                key_word=['amount'], freq='D', index=1, ret_index=False)
vwap_amount['Fut_code'] = vwap_amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
vwap_amount['bpv'] = vwap_amount['Fut_code'].map(ConstFutBasic.fut_code_bpv)
vwap_amount['amount'] = vwap_amount['amount'] / vwap_amount['bpv'] * 10000
vwap_amount = vwap_amount.pivot_table(index='Trade_DT', columns='Fut_code', values=vwap_amount.columns[2])
vwap_volume = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                key_word='vol', freq='D', index=1, ret_index=False)
vwap_volume['Fut_code'] = vwap_volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
vwap_volume = vwap_volume.pivot_table(index='Trade_DT', columns='Fut_code', values=vwap_volume.columns[2])

vwap = vwap_amount / vwap_amount
vwap = vwap.iloc[:-1, ].reset_index(drop=True)

amount = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='amount', freq='D', index=1, ret_index=False)
amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])
amount = amount.iloc[:-1, ].reset_index(drop=True)

carry_pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                              key_word='annualized_carry_rate_nxt_main', freq='D', index=1,
                                              ret_index=False)
carry_pct['Fut_code'] = carry_pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
carry_pct = carry_pct.pivot_table(index='Trade_DT', columns='Fut_code', values=carry_pct.columns[2])
carry_pct = carry_pct.iloc[:-1, ].reset_index(drop=True)

historical_vol = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='historical_volatility', freq='D', index=1, ret_index=False)
historical_vol['Fut_code'] = historical_vol['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
historical_vol = historical_vol.pivot_table(index='Trade_DT', columns='Fut_code', values=historical_vol.columns[2])
historical_vol = historical_vol.iloc[:-1, ].reset_index(drop=True)

dastd = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='dastd', freq='D', index=1, ret_index=False)
dastd['Fut_code'] = dastd['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
dastd = dastd.pivot_table(index='Trade_DT', columns='Fut_code', values=dastd.columns[2])
dastd = dastd.iloc[:-1, ].reset_index(drop=True)

pct1 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='close', freq='D', index=1, ret_index=True)
pct1['Fut_code'] = pct1['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct1 = pct1.pivot_table(index='Trade_DT', columns='Fut_code', values=pct1.columns[2])
pct1 = pct1.iloc[:-1, :].reset_index(drop=True)

pct2 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq='D', index=1, ret_index=True)
pct2['Fut_code'] = pct2['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct2 = pct2.pivot_table(index='Trade_DT', columns='Fut_code', values=pct2.columns[2])
pct2 = pct2.iloc[:-1, :].reset_index(drop=True)

pct3 = high.div(low) - 1
pct4 = close.div(open) - 1

##################################
# GA
#################################

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(pd.DataFrame, 16), pd.DataFrame)


def rollz(df, N):
    return (df.sub(df.rolling(N).mean())).div(df.rolling(N).std())


pset.addPrimitive(rollz, [pd.DataFrame, int], pd.DataFrame)


def rollsum(df, N):
    return df.rolling(N).sum()


pset.addPrimitive(rollsum, [pd.DataFrame, int], pd.DataFrame)


def rollmax(df, N):
    return df.rolling(N).max()


pset.addPrimitive(rollmax, [pd.DataFrame, int], pd.DataFrame)


def rollmin(df, N):
    return df.rolling(N).min()


pset.addPrimitive(rollmin, [pd.DataFrame, int], pd.DataFrame)


def add(left, right):
    return left.add(right)


pset.addPrimitive(add, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def sub(left, right):
    return left.sub(right)


pset.addPrimitive(sub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def div(left, right):
    return left.div(right)


pset.addPrimitive(div, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def mul(left, right):
    return left.mul(right)


pset.addPrimitive(mul, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def absdiv(left, right):
    return left.div(abs(right))


pset.addPrimitive(absdiv, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def rollma(df, N):
    return df.rolling(N).mean()


pset.addPrimitive(rollma, [pd.DataFrame, int], pd.DataFrame)


def changeparam(df, N):
    return df.rolling(N).std().div(abs(df.rolling(N).mean()))


pset.addPrimitive(changeparam, [pd.DataFrame, int], pd.DataFrame)


def rollstd(df, N):
    return df.rolling(N).std()


pset.addPrimitive(rollstd, [pd.DataFrame, int], pd.DataFrame)


def tsmeanbias(df, N):
    return df.sub(df.rolling(N).mean())


pset.addPrimitive(tsmeanbias, [pd.DataFrame, int], pd.DataFrame)


def ma2biasvalue(df, N1, N2):
    ma1 = (df.rolling(N1).mean()).div(df.shift(N1))
    ma2 = (df.rolling(N2).mean()).div(df.shift(N2))
    return ma1.sub(ma2)


pset.addPrimitive(ma2biasvalue, [pd.DataFrame, int, int], pd.DataFrame)


def mabiaspct(df, N):
    return df.div(df.rolling(N).mean())


pset.addPrimitive(mabiaspct, [pd.DataFrame, int], pd.DataFrame)


def maabsbiaspct(df, N):
    return df.div(abs(df.rolling(N).mean()))


pset.addPrimitive(maabsbiaspct, [pd.DataFrame, int], pd.DataFrame)


def ma2biaspct(df, N1, N2):
    ma1 = df.rolling(N1).mean()
    ma2 = df.rolling(N2).mean()
    return ma1.div(ma2)


pset.addPrimitive(ma2biaspct, [pd.DataFrame, int, int], pd.DataFrame)


def ts_diff_lag1(data):
    return data.sub(data.shift(1))


pset.addPrimitive(ts_diff_lag1, [pd.DataFrame], pd.DataFrame)


def tslag1std(data, N):
    return (data.sub(data.shift(1))).div(data.rolling(N).std())


pset.addPrimitive(tslag1std, [pd.DataFrame, int], pd.DataFrame)


def tslag1mean(data, N):
    return (data.sub(data.shift(1))).div(data.rolling(N).mean())


pset.addPrimitive(tslag1mean, [pd.DataFrame, int], pd.DataFrame)


def corr(left, right, N):
    return left.rolling(N).corr(right)


pset.addPrimitive(corr, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)


def cov(left, right, N):
    return left.rolling(N).corr(right).mul(left.rolling(N).std()).mul(right.rolling(N).std())


pset.addPrimitive(cov, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)


def zcorr(left, right, N1, N2):
    return corr(rollz(left, N1), rollz(right, N1), N2)


pset.addPrimitive(zcorr, [pd.DataFrame, pd.DataFrame, int, int], pd.DataFrame)


def deltacorr(df, N):
    return df.rolling(N).corr(df.sub(df.shift(1)))


pset.addPrimitive(deltacorr, [pd.DataFrame, int], pd.DataFrame)


def lag1_rb(df):
    data = df.sub(df.shift(1))
    return (data.sub(data.shift(1))).div(abs(data).shift(1))


pset.addPrimitive(lag1_rb, [pd.DataFrame], pd.DataFrame)


def lag1(df):
    return df.shift(1)


pset.addPrimitive(lag1, [pd.DataFrame], pd.DataFrame)


def lagN(df, N):
    return df.shift(N)


pset.addPrimitive(lagN, [pd.DataFrame, int], pd.DataFrame)


def skew(data, N):
    return data.rolling(N).skew()


pset.addPrimitive(skew, [pd.DataFrame, int], pd.DataFrame)


def kurt(data, N):
    return data.rolling(N).kurt()


pset.addPrimitive(kurt, [pd.DataFrame, int], pd.DataFrame)


def org(data):
    return data


pset.addPrimitive(org, [pd.DataFrame], pd.DataFrame)


def neg(data):
    return data * -1


pset.addPrimitive(neg, [pd.DataFrame], pd.DataFrame)


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


def int_5():
    return 5


pset.addPrimitive(int_5, [], int)


def int_9():
    return 9


pset.addPrimitive(int_9, [], int)


def int_10():
    return 10


pset.addPrimitive(int_10, [], int)


def int_11():
    return 11


pset.addPrimitive(int_11, [], int)


def int_13():
    return 13


pset.addPrimitive(int_13, [], int)


def int_14():
    return 14


pset.addPrimitive(int_14, [], int)


def int_20():
    return 20


pset.addPrimitive(int_20, [], int)


def int_21():
    return 21


pset.addPrimitive(int_21, [], int)

pset.addEphemeralConstant("const1", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const2", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const3", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const4", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const5", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const7", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const9", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const10", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const11", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const13", lambda: random.randint(1, 25), int)

pset.renameArguments(ARG0='close')
pset.renameArguments(ARG1='open')
pset.renameArguments(ARG2='high')
pset.renameArguments(ARG3='low')
pset.renameArguments(ARG4='settle')
pset.renameArguments(ARG5='volume')
pset.renameArguments(ARG6='oi')
pset.renameArguments(ARG7='vwap')
pset.renameArguments(ARG8='amount')
pset.renameArguments(ARG9='carry_pct')
pset.renameArguments(ARG10='historical_vol')
pset.renameArguments(ARG11='dastd')
pset.renameArguments(ARG12='pct1')
pset.renameArguments(ARG13='pct2')
pset.renameArguments(ARG14='pct3')
pset.renameArguments(ARG15='pct4')

# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
# toolbox = base.Toolbox()
#
# toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
# # 考虑初值生成方式
#
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", gp.compile, pset=pset)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


def evalSymbReg(individual, pct, close, open, high, low, settle, volume, oi, vwap, amount, carry_pct, historical_vol,
                dastd, pct1, pct2, pct3, pct4):
    function = toolbox.compile(expr=individual)
    res = function(close, open, high, low, settle, volume, oi, vwap, amount, carry_pct, historical_vol, dastd, pct1,
                   pct2, pct3, pct4)
    res1 = res.replace([np.inf, -np.inf], np.nan)
    res1 = res1.where(pct.isnull() == False).dropna(axis=0, how='all')

    cor = res1.corrwith(pct, axis=0).dropna()
    # res_mean = cor.mean()
    # res_std = cor.std()
    try:
        result = cor.abs().sum()
        if result < np.inf:
            return result,
        else:
            return -np.inf,
    except:
        return -np.inf,

    # if (len(cor) < len(pct) * 0.9) or (res_mean == 0) or (res_std == 0):
    #     return -np.inf,
    # else:
    #     cot = res1.count()
    #     if len(cot.where(cot <= len(res1) * 0.9).dropna()) >= len(res1.columns) * 0.5:
    #         return -np.inf,
    #     else:
    #         return abs(res_mean),


def evalSymbReg_return(individual, pct, close, open, high, low, settle, volume, oi, vwap, amount, carry_pct,
                       historical_vol,
                       dastd, pct1, pct2, pct3, pct4):
    function = toolbox.compile(expr=individual)
    res = function(close, open, high, low, settle, volume, oi, vwap, amount, carry_pct, historical_vol, dastd, pct1,
                   pct2, pct3, pct4)
    res1 = res.where(pct.isnull() == False).dropna(axis=0, how='all')
    pct_calculated = pct.where(pct.isnull() == False).dropna(axis=0, how='all')
    res1 = res1.applymap(lambda x: 1 if x > 0 else 0)
    result = res1 * pct_calculated
    result = result.sum().mean()

    # cor = res1.corrwith(pct, axis=0).dropna()
    # res_mean = cor.mean()
    # res_std = cor.std()
    try:
        # result = cor.mean() + cor.max()
        if result < np.inf:
            return result,
        else:
            return -np.inf,
    except:
        return -np.inf,

    # if (len(cor) < len(pct) * 0.9) or (res_mean == 0) or (res_std == 0):
    #     return -np.inf,
    # else:
    #     cot = res1.count()
    #     if len(cot.where(cot <= len(res1) * 0.9).dropna()) >= len(res1.columns) * 0.5:
    #         return -np.inf,
    #     else:
    #         return abs(res_mean),


toolbox.register("evaluate", evalSymbReg, pct=pct, close=close, open=open, high=high, low=low, settle=settle,
                 volume=volume, oi=oi, vwap=vwap, amount=amount, carry_pct=carry_pct, historical_vol=historical_vol,
                 dastd=dastd, pct1=pct1, pct2=pct2, pct3=pct3, pct4=pct4)
# toolbox.register("select", tools.selBest)  # select数量
toolbox.register("select", tools.selTournament, tournsize=4)  # select数量
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def main():
    pop = toolbox.population(n=5000)
    #    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats用来汇报结果
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # mstats.register("avg", np.nanmean)
    #    mstats.register("std", np.nanstd)
    #    mstats.register("min", np.nanmin)
    mstats.register("max", np.nanmax)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=1, stats=mstats,  # halloffame=hof,
                                   verbose=True)
    # print log
    return pop, log  # , hof


if __name__ == '__main__':
    print(datetime.datetime.now())
    # pool = Pool(5)
    # toolbox.register("map", pool.map)
    print("start_programming")
    pop, log = main()
    top = tools.selBest(pop, k=200)
    noDupes = []
    [noDupes.append(i) for i in top if not noDupes.count(i)]
    print(datetime.datetime.now())
    for i in range(len(noDupes)):
        tree = gp.PrimitiveTree(noDupes[i])
        print(str(tree))
        function = gp.compile(tree, pset)
        res = function(close, open, high, low, settle, volume, oi, vwap, amount, carry_pct, historical_vol, dastd, pct1,
                       pct2, pct3, pct4)

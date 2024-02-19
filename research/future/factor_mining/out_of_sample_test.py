import os
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
end_date = '2022-09-16'
fee = 0.0005
out_of_sample_size = 0

# basic price volume factor


close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq='D', index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close = close.iloc[:-1, ]
close_all = close.copy()
close = close_all.iloc[-1 * out_of_sample_size:, ]

index_all = close_all.index
index_test = close.index

close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq='D', index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close = close.iloc[:-1, ]
close.index = range(len(close))
close_all = close.copy()
close = close_all.iloc[-1 * out_of_sample_size:, ]

open = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq='D', index=1, ret_index=False)
open['Fut_code'] = open['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open = open.pivot_table(index='Trade_DT', columns='Fut_code', values=open.columns[2])
open = open.iloc[:-1, ]
open.index = range(len(open))
open_all = open.copy()
open = open_all.iloc[-1 * out_of_sample_size:, ]

high = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='high', freq='D', index=1, ret_index=False)
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
                                           key_word='amount', freq='D', index=1, ret_index=False)
amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])
amount = amount.iloc[:-1, ].reset_index(drop=True)
amount_all = amount.copy()
amount = amount_all.iloc[-1 * out_of_sample_size:, ]

open_pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                             key_word='open', freq='D', index=1, ret_index=False)
open_pct['Fut_code'] = open_pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open_pct = open_pct.pivot_table(index='Trade_DT', columns='Fut_code', values=open_pct.columns[2])
close_pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                              key_word='close', freq='D', index=1, ret_index=False)
close_pct['Fut_code'] = close_pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close_pct = close_pct.pivot_table(index='Trade_DT', columns='Fut_code', values=close_pct.columns[2])
ret = close_pct * (1 - fee) / open_pct * (1 + fee) - 1
ret = ret.iloc[1:, ].reset_index(drop=True)

ret_all = ret.copy()
ret = ret_all.iloc[-1 * out_of_sample_size:, ]

I = pd.DataFrame(1, index=ret.index, columns=ret.columns)

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(pd.DataFrame, 1), pd.DataFrame)


# 同时两个信号
def both(signal1, signal2):
    return I.where(signal1.add(signal2) >= 2, 0)


pset.addPrimitive(both, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def either(signal1, signal2):
    return I.where(signal1.add(signal2) >= 1, 0)


pset.addPrimitive(either, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def both3(signal1, signal2, signal3):
    return I.where(signal1.add(signal2).add(signal3) >= 3, 0)


pset.addPrimitive(both3, [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)


def both4(signal1, signal2, signal3, signal4):
    return I.where(signal1.add(signal2).add(signal3).add(signal4) >= 4, 0)


pset.addPrimitive(both4, [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)


def both5(signal1, signal2, signal3, signal4, signal5):
    return I.where(signal1.add(signal2).add(signal3).add(signal4).add(signal5) >= 5, 0)


pset.addPrimitive(both5, [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)


# 收于均线下方
def closeoverma(N):
    ma = close.rolling(N).mean()
    sig = I.where(close > ma, 0)
    return sig


pset.addPrimitive(closeoverma, [int], pd.DataFrame)


# 收于均线上方
def closebelowma(N):
    ma = close.rolling(N).mean()
    sig = I.where(close < ma, 0)
    return sig


pset.addPrimitive(closebelowma, [int], pd.DataFrame)


# 均线上穿
def ma1overma2(N1, N2):
    ma1 = close.rolling(N1).mean()
    ma2 = close.rolling(N2).mean()
    sig = I.where(ma1 > ma2, 0)
    return sig


pset.addPrimitive(ma1overma2, [int, int], pd.DataFrame)


# 长上影线
def longupshadow(pct):
    maxop = open.where(open > close, close)
    upshadow = high.sub(maxop)
    sig = I.where(upshadow.div(high.sub(low)) > pct, 0)
    return sig


pset.addPrimitive(longupshadow, [float], pd.DataFrame)


# N天K长上影线
def longupshadowN(pct, N):
    openn = open.shift(N)
    highn = high.rolling(N).max()
    lown = low.rolling(N).min()
    maxop = openn.where(openn > close, close)
    upshadow = highn.sub(maxop)
    sig = I.where(upshadow.div(highn.sub(lown)) > pct, 0)
    return sig


pset.addPrimitive(longupshadowN, [float, int], pd.DataFrame)


# 长下影线
def longdownshadow(pct):
    minop = open.where(open < close, close)
    downshadow = minop.sub(low)
    sig = I.where(downshadow.div(high.sub(low)) > pct, 0)
    return sig


pset.addPrimitive(longdownshadow, [int], pd.DataFrame)


# N天长下影线
def longdownshadowN(pct, N):
    openn = open.shift(N)
    highn = high.rolling(N).max()
    lown = low.rolling(N).min()
    minop = openn.where(openn < close, close)
    downshadow = minop.sub(lown)
    sig = I.where(downshadow.div(highn.sub(lown)) > pct, 0)
    return sig


pset.addPrimitive(longdownshadowN, [float, int], pd.DataFrame)


# 短上影线
def shortupshadow(pct):
    maxop = open.where(open > close, close)
    upshadow = high.sub(maxop)
    sig = I.where(upshadow.div(high.sub(low)) < pct, 0)
    return sig


pset.addPrimitive(shortupshadow, [float], pd.DataFrame)


# N天短上影线
def shortupshadowN(pct, N):
    openn = open.shift(N)
    highn = high.rolling(N).max()
    lown = low.rolling(N).min()
    maxop = openn.where(openn > close, close)
    upshadow = highn.sub(maxop)
    sig = I.where(upshadow.div(highn.sub(lown)) < pct, 0)
    return sig


pset.addPrimitive(shortupshadowN, [float, int], pd.DataFrame)


# 短下影线
def shortdownshadow(pct):
    minop = open.where(open < close, close)
    downshadow = minop.sub(low)
    sig = I.where(downshadow.div(high.sub(low)) < pct, 0)
    return sig


pset.addPrimitive(shortdownshadow, [float], pd.DataFrame)


# N天短下影线
def shortdownshadowN(pct, N):
    openn = open.shift(N)
    highn = high.rolling(N).max()
    lown = low.rolling(N).min()
    minop = openn.where(openn < close, close)
    downshadow = minop.sub(lown)
    sig = I.where(downshadow.div(highn.sub(lown)) < pct, 0)
    return sig


pset.addPrimitive(shortdownshadowN, [float, int], pd.DataFrame)


# 十字星
def crossstar(pct):
    tang = abs(open.sub(close))
    alllen = high.sub(low)
    sig = I.where(tang.div(alllen) < pct, 0)
    return sig


pset.addPrimitive(crossstar, [float], pd.DataFrame)


# N天十字星
def crossstarN(pct, N):
    openn = open.shift(N)
    highn = high.rolling(N).max()
    lown = low.rolling(N).min()
    tang = abs(openn.sub(close))
    alllen = highn.sub(lown)
    sig = I.where(tang.div(alllen) < pct, 0)
    return sig


pset.addPrimitive(crossstarN, [float, int], pd.DataFrame)


# 盘中突破N天新高
def highnewhigh(N):
    sig = I.where(high >= high.rolling(N).max(), 0)
    return sig


pset.addPrimitive(highnewhigh, [int], pd.DataFrame)


# 收盘突破N天新高
def closenewhigh(N):
    sig = I.where(close >= close.rolling(N).max(), 0)
    return sig


pset.addPrimitive(closenewhigh, [int], pd.DataFrame)


# 盘中突破N天新低
def lownewlow(N):
    sig = I.where(low <= low.rolling(N).min(), 0)
    return sig


pset.addPrimitive(lownewlow, [int], pd.DataFrame)


# 收盘突破N天新低
def closenewlow(N):
    sig = I.where(close <= close.rolling(N).min(), 0)
    return sig


pset.addPrimitive(closenewlow, [int], pd.DataFrame)


# 开盘跳空低开低于N天最低
def openjumplowN(N):
    histlow = low.rolling(N).min().shift(1)
    sig = I.where(open < histlow, 0)
    return sig


pset.addPrimitive(openjumplowN, [int], pd.DataFrame)


# 开盘跳空低开低于前一天天最低
def openjumplow():
    histlow = low.shift(1)
    sig = I.where(open < histlow, 0)
    return sig


pset.addPrimitive(openjumplow, [], pd.DataFrame)


# 收盘调控低开低于N天最低
def closejumplowN(N):
    histlow = low.rolling(N).min().shift(1)
    sig = I.where(close < histlow, 0)
    return sig


pset.addPrimitive(closejumplowN, [int], pd.DataFrame)


# 收盘调控低开低于前一天最低
def closejumplow():
    histlow = low.shift(1)
    sig = I.where(close < histlow, 0)
    return sig


pset.addPrimitive(closejumplow, [], pd.DataFrame)


# 开盘跳空高开
def openjumphighN(N):
    histhigh = high.rolling(N).max().shift(1)
    sig = I.where(open > histhigh, 0)
    return sig


pset.addPrimitive(openjumphighN, [int], pd.DataFrame)


# 开盘跳空高开
def openjumphigh():
    histhigh = high.shift(1)
    sig = I.where(open > histhigh, 0)
    return sig


pset.addPrimitive(openjumphigh, [], pd.DataFrame)


# 收盘跳空高开
def closejumphighN(N):
    histhigh = high.rolling(N).max().shift(1)
    sig = I.where(close > histhigh, 0)
    return sig


pset.addPrimitive(closejumphighN, [int], pd.DataFrame)


# 收盘跳空高开
def closejumphigh():
    histhigh = high.shift(1)
    sig = I.where(close > histhigh, 0)
    return sig


pset.addPrimitive(closejumphigh, [], pd.DataFrame)


# 开盘价高开
def openhigher():
    return I.where(open > close.shift(1), 0)


pset.addPrimitive(openhigher, [], pd.DataFrame)


# 开盘价低开
def openlower():
    return I.where(open < close.shift(1), 0)


pset.addPrimitive(openlower, [], pd.DataFrame)


# 前N天的信号
def lag(signal, N):
    return signal.shift(N)


pset.addPrimitive(lag, [pd.DataFrame, int], pd.DataFrame)


# 反向指标
def neg(signal, N):
    return I.where(signal < 1, 0)


pset.addPrimitive(neg, [pd.DataFrame, int], pd.DataFrame)


# 实体波动大于过去N天均值
def ocgrtNmean(N):
    Nmean = abs(close.div(open) - 1).rolling(N).mean().shift(1)
    return I.where(abs(close.div(open) - 1) > Nmean, 0)


pset.addPrimitive(ocgrtNmean, [int], pd.DataFrame)


# 实体波动大于过去N天最大值
def ocgrtNmax(N):
    Nmax = abs(close.div(open) - 1).rolling(N).max().shift(1)
    return I.where(abs(close.div(open) - 1) > Nmax, 0)


pset.addPrimitive(ocgrtNmax, [int], pd.DataFrame)


# 实体波动小于过去N天最小值
def oclsNmin(N):
    Nmin = abs(close.div(open) - 1).rolling(N).min().shift(1)
    return I.where(abs(close.div(open) - 1) < Nmin, 0)


pset.addPrimitive(oclsNmin, [int], pd.DataFrame)


# 影线波动大于过去N天均值
def shgrtNmean(N):
    Nmean = abs(high.div(low) - 1).rolling(N).mean().shift(1)
    return I.where(abs(high.div(low) - 1) > Nmean, 0)


pset.addPrimitive(shgrtNmean, [int], pd.DataFrame)


# 影线波动大于过去N天最大值
def shgrtNmax(N):
    Nmax = abs(high.div(low) - 1).rolling(N).max().shift(1)
    return I.where(abs(high.div(low) - 1) > Nmax, 0)


pset.addPrimitive(shgrtNmax, [int], pd.DataFrame)


# 影线波动小于过去N天最小值
def shlsNmin(N):
    Nmin = abs(high.div(low) - 1).rolling(N).min().shift(1)
    return I.where(abs(high.div(low) - 1) < Nmin, 0)


pset.addPrimitive(shlsNmin, [int], pd.DataFrame)


# 过去N天出现过信号
def everpositive(signal, N):
    return I.where(signal.rolling(N).sum() > 0, 0)


pset.addPrimitive(everpositive, [pd.DataFrame, int], pd.DataFrame)


# 过去N天出现信号大于阈值
def positivenum(signal, N1, N2):
    return I.where(signal.rolling(N1).sum() > N2)


pset.addPrimitive(positivenum, [pd.DataFrame, int, int], pd.DataFrame)


# 过去N天没出现过信号
def neverpositive(signal, N):
    return I.where(signal.rolling(N).sum() < 1, 0)


pset.addPrimitive(neverpositive, [pd.DataFrame, int], pd.DataFrame)


# 收盘是最低价
def closelowest():
    return I.where(close <= low, 0)


pset.addPrimitive(closelowest, [], pd.DataFrame)


# 收盘是最高价
def closehighest():
    return I.where(close >= high, 0)


pset.addPrimitive(closehighest, [], pd.DataFrame)


# 开盘是最低价
def openlowest():
    return I.where(open <= low, 0)


pset.addPrimitive(openlowest, [], pd.DataFrame)


# 开盘是最高价
def openhighest():
    return I.where(open >= high, 0)


pset.addPrimitive(openhighest, [], pd.DataFrame)


# 当日上涨
def priceup():
    return I.where(close > close.shift(1), 0)


pset.addPrimitive(priceup, [], pd.DataFrame)


# N天上涨
def Ndayup(N):
    return I.where(close > close.shift(N), 0)


pset.addPrimitive(Ndayup, [int], pd.DataFrame)


# 红K线
def redK():
    return I.where(close > open, 0)


pset.addPrimitive(redK, [], pd.DataFrame)


# N天K线红
def redKN(N):
    return I.where(close > open.shift(N), 0)


pset.addPrimitive(redKN, [int], pd.DataFrame)


# 绿K线
def greenK():
    return I.where(close < open, 0)


pset.addPrimitive(greenK, [], pd.DataFrame)


# N天K线绿
def greenKN(N):
    return I.where(close < open.shift(N), 0)


pset.addPrimitive(greenKN, [int], pd.DataFrame)


# N天位移路程比超过阈值
def closemovedistratio(pct, N):
    move = abs(close.div(close.shift(N)) - 1)
    distance = abs(close.div(close.shift(1)) - 1).rolling(N).sum()
    return I.where(move.div(distance) > pct, 0)


pset.addPrimitive(closemovedistratio, [float, int], pd.DataFrame)


# K线实体相对前一日长
def Klonger():
    Klen = abs(close.div(open) - 1)
    return I.where(Klen > Klen.shift(1), 0)


pset.addPrimitive(Klonger, [], pd.DataFrame)


# K线实体相对前一日短
def Kshorter():
    Klen = abs(close.div(open) - 1)
    return I.where(Klen < Klen.shift(1), 0)


pset.addPrimitive(Kshorter, [], pd.DataFrame)


# N天K线实体相对前一期长
def NKlonger(N):
    NKlen = abs(close.div(open.shift(N)) - 1)
    return I.where(NKlen > NKlen.shift(N + 1), 0)


pset.addPrimitive(NKlonger, [int], pd.DataFrame)


# N天K线实体相对前一期短
def NKshorter(N):
    NKlen = abs(close.div(open.shift(N)) - 1)
    return I.where(NKlen < NKlen.shift(N + 1), 0)


pset.addPrimitive(NKshorter, [int], pd.DataFrame)


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


def int_5():
    return 5


pset.addPrimitive(int_5, [], int)


def int_7():
    return 7


pset.addPrimitive(int_7, [], int)


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


def int_17():
    return 17


pset.addPrimitive(int_17, [], int)


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


def float_025():
    return 0.25


pset.addPrimitive(float_025, [], float)


def float_03():
    return 0.3


pset.addPrimitive(float_03, [], float)


def float_035():
    return 0.35


pset.addPrimitive(float_035, [], float)


def float_04():
    return 0.4


pset.addPrimitive(float_04, [], float)


def float_045():
    return 0.45


pset.addPrimitive(float_045, [], float)


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

pset.addEphemeralConstant("const1", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const2", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const3", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const4", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const5", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const7", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const9", lambda: random.randint(1, 25), int)
pset.addEphemeralConstant("const10", lambda: random.randint(1, 25), int)

pset.addEphemeralConstant("float1", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float2", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float3", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float4", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float5", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float6", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float7", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float8", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float9", lambda: random.uniform(0, 1), float)
pset.addEphemeralConstant("float10", lambda: random.uniform(0, 1), float)

pset.renameArguments(ARG0='I')
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=10)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, ret, I, amount):
    function = toolbox.compile(expr=individual)
    res = function(I)
    res1 = res.where(ret.isnull() == False).dropna(axis=0, how='all')
    amountrank = (-1 * amount).where(res1 > 0).rank(axis=1)

    ret_cond = ret.where(amountrank <= 10)
    ret_mean = ret_cond.sum(axis=1) / 10
    if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
        return -np.inf, -np.inf,
    else:
        return ret_mean.dropna().mean() / ret_mean.dropna().std(), ret_mean.fillna(0).sum(),


if __name__ != "__main__":
    file_path = 'C:\\Users\\jason.huang\\research\\data_mining\\pattern_technical_indicator\\'
    save_path = 'C:\\Users\\jason.huang\\research\\data_mining\\pattern_technical_indicator_out_of_sample\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file_name in [x for x in os.listdir(file_path) if x not in os.listdir(save_path)]:
        file_location = file_path + file_name
        data = pd.read_csv(file_location, index_col=0)

        save_location = save_path + file_name
        result_all = pd.DataFrame()
        for f in data.columns:
            try:
                res = eval(f)
                res1 = res.where(ret_all.isnull() == False).dropna(axis=0, how='all')
                amountrank = (-1 * amount).where(res1 > 0).rank(axis=1)
                ret_cond = ret_all.where(amountrank <= 10)
                ret_mean = ret_cond.sum(axis=1) / 10
                cum_pnl = ret_mean.cumsum() + 1
                cum_pnl = pd.DataFrame(cum_pnl)
                cum_pnl.index = index_test
                cum_pnl.columns = [f]
                result_all = pd.concat([result_all, cum_pnl], axis=1)
                result_all.to_csv(save_location)
            except Exception as e:
                pass

    file_path = 'C:\\Users\\jason.huang\\research\\data_mining\\pattern_technical_indicator\\'
    col_list=[]
    for file_name in os.listdir(file_path):
        file_location = file_path + file_name
        data = pd.read_csv(file_location, index_col=0)
        print(data.shape)
        col_list.append(data.columns)
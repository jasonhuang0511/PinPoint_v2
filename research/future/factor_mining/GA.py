
import multiprocessing
import statsmodels.api as sm
import numpy as np
import pandas as pd
import operator
import itertools
import time
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pymssql
import datetime
import random
import math

import warnings

warnings.filterwarnings('ignore')

file_location='C:\\Users\\admin\\Desktop\\笔试(含数据)\\test_df.csv'
data=pd.read_csv(file_location)

open_price=data.pivot_table(values='open',index='time',columns='asset').sort_index()
high_price=data.pivot_table(values='high',index='time',columns='asset').sort_index()
low_price=data.pivot_table(values='low',index='time',columns='asset').sort_index()
close_price=data.pivot_table(values='close',index='time',columns='asset').sort_index()
volume=data.pivot_table(values='volume',index='time',columns='asset').sort_index()
volumeU=data.pivot_table(values='volumeU',index='time',columns='asset').sort_index()
volume_takerBuy=data.pivot_table(values='volume_takerBuy',index='time',columns='asset').sort_index()
volumeU_takerBuy=data.pivot_table(values='volumeU_takerBuy',index='time',columns='asset').sort_index()
trades=data.pivot_table(values='trades',index='time',columns='asset').sort_index()

# 预测收益
pct=close_price.div(close_price.shift(1))-1
pct=pct.iloc[1:,]
pct.index=range(len(pct))

# 基础因子
pct1=close_price.div(close_price.shift(1))-1
pct1=pct1.iloc[:-1,]
pct1.index=range(len(pct1))

pct2=open_price.div(open_price.shift(1))-1
pct2=pct2.iloc[:-1,]
pct2.index=range(len(pct2))

pct3=high_price.div(low_price)-1
pct3=pct3.iloc[:-1,]
pct3.index=range(len(pct3))

pct4=close_price.div(open_price)-1
pct4=pct4.iloc[:-1,]
pct4.index=range(len(pct4))

open_price=open_price.iloc[:-1,]
open_price.index=range(len(open_price))

high_price=high_price.iloc[:-1,]
high_price.index=range(len(high_price))


low_price=low_price.iloc[:-1,]
low_price.index=range(len(low_price))

close_price=close_price.iloc[:-1,]
close_price.index=range(len(close_price))

volume=volume.iloc[:-1,]
volume.index=range(len(volume))

volumeU=volumeU.iloc[:-1,]
volumeU.index=range(len(volumeU))

volume_takerBuy=volume_takerBuy.iloc[:-1,]
volume_takerBuy.index=range(len(volume_takerBuy))

volumeU_takerBuy=volumeU_takerBuy.iloc[:-1,]
volumeU_takerBuy.index=range(len(volumeU_takerBuy))

trades=trades.iloc[:-1,]
trades.index=range(len(trades))


pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(pd.DataFrame, 14), pd.DataFrame)


# 日间操作
def zscore(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)


pset.addPrimitive(zscore, [pd.DataFrame], pd.DataFrame)


def rollz(df, N):
    return (df.sub(df.rolling(N).mean())).div(df.rolling(N).std())


pset.addPrimitive(rollz, [pd.DataFrame, int], pd.DataFrame)


def rank(df):
    return df.rank(1)


pset.addPrimitive(rank, [pd.DataFrame], pd.DataFrame)


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


def rollma(data, N):
    return data.rolling(N).mean()


pset.addPrimitive(rollma, [pd.DataFrame, int], pd.DataFrame)


def changeparam(data, N):
    return data.rolling(N).std().div(abs(data.rolling(N).mean()))


pset.addPrimitive(changeparam, [pd.DataFrame, int], pd.DataFrame)


def rollstd(data, N):
    return data.rolling(N).std()


pset.addPrimitive(rollstd, [pd.DataFrame, int], pd.DataFrame)


def tsmeanbias(data, N):
    return data.sub(data.rolling(N).mean())


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


def tslag1(data):
    return data.sub(data.shift(1))


pset.addPrimitive(tslag1, [pd.DataFrame], pd.DataFrame)


def tslag1std(data, N):
    return (data.sub(data.shift(1))).div(data.rolling(N).std())


pset.addPrimitive(tslag1std, [pd.DataFrame, int], pd.DataFrame)


def tslag1mean(data, N):
    return (data.sub(data.shift(1))).div(data.rolling(N).mean())


pset.addPrimitive(tslag1mean, [pd.DataFrame, int], pd.DataFrame)



def zadd(left, right):
    return zscore(left).add(zscore(right))


pset.addPrimitive(zadd, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def zsub(left, right):
    return zscore(left).sub(zscore(right))


pset.addPrimitive(zsub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def corr(left, right, N):
    return left.rolling(N).corr(right)


pset.addPrimitive(corr, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)


def cov(left, right, N):
    return left.rolling(N).corr(right).mul(left.rolling(N).std()).mul(right.rolling(N).std())


pset.addPrimitive(cov, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)


def zcorr(left, right, N1, N2):
    return corr(rollz(left, N1), rollz(right, N1), N2)


pset.addPrimitive(zcorr, [pd.DataFrame, pd.DataFrame, int, int], pd.DataFrame)


def meanzaddlast(df, N):
    return zadd(rollma(df, N), df)


pset.addPrimitive(meanzaddlast, [pd.DataFrame, int], pd.DataFrame)


def deltacorr(df, N):
    return df.rolling(N).corr(df.sub(df.shift(1)))


pset.addPrimitive(deltacorr, [pd.DataFrame, int], pd.DataFrame)


def lag1_rb(data):
    df = data.sub(data.shift(1))
    return (df.sub(df.shift(1))).div(abs(df).shift(1))


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

pset.renameArguments(ARG0='pct')
pset.renameArguments(ARG1='pct1')
pset.renameArguments(ARG2='pct2')
pset.renameArguments(ARG3='pct3')
pset.renameArguments(ARG4='pct4')
pset.renameArguments(ARG5='open_price')
pset.renameArguments(ARG6='high_price')
pset.renameArguments(ARG7='low_price')
pset.renameArguments(ARG8='close_price')
pset.renameArguments(ARG9='volume')
pset.renameArguments(ARG10='volumeU')
pset.renameArguments(ARG11='volume_takerBuy')
pset.renameArguments(ARG12='volumeU_takerBuy')
pset.renameArguments(ARG13='trades')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
# 考虑初值生成方式

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


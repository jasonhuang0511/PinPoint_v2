# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:49:19 2020

@author: user
"""

# from multiprocessing import Pool
from multiprocessing.dummy import Pool
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import bottleneck as bn
import operator
import itertools
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import datetime
import sys

sys.path.append('E:\\ICmining')
# from pypdf import reporting
import random
# import math
# import gc
import warnings

warnings.filterwarnings('ignore')

dirct = "E:\\ZZ500mindata\\"
# 原始数据导入
amount = pd.read_pickle(dirct + 'zz500amount20200101_20200930').sort_index().query("time not in (1457,1458,1459,1500)")
benchopen = pd.read_pickle(dirct + 'zz500benchmarkopen20200101_20200930').sort_index().query(
    "time not in (1457,1458,1459,1500)")
closep = pd.read_pickle(dirct + 'zz500closeprice20200101_20200930').sort_index().query(
    "time not in (1457,1458,1459,1500)")
# cumhigh = pd.read_pickle(dirct + 'zz500cumhigh20200101_20200930').sort_index()
# cumlow = pd.read_pickle(dirct + 'zz500cumlow20200101_20200930').sort_index()
# cumvol = pd.read_pickle(dirct + 'zz500cumvolume20200101_20200930').sort_index()
high = pd.read_pickle(dirct + 'zz500high20200101_20200930').sort_index().query("time not in (1457,1458,1459,1500)")
low = pd.read_pickle(dirct + 'zz500low20200101_20200930').sort_index().query("time not in (1457,1458,1459,1500)")
openp = pd.read_pickle(dirct + 'zz500openprice20200101_20200930').sort_index().query(
    "time not in (1457,1458,1459,1500)")
pct = pd.read_pickle(dirct + 'zz500pct20200101_20200930').sort_index().query("time not in (1457,1458,1459,1500)")
vol = pd.read_pickle(dirct + 'zz500volume20200101_20200930').sort_index().query("time not in (1457,1458,1459,1500)")
vwap = pd.read_pickle(dirct + 'zz500vwap20200101_20200930').sort_index().query("time not in (1457,1458,1459,1500)")

hlp = high.sub(low)
hlr = high.div(low) - 1
ocp = closep.sub(openp)
ocr = closep.div(openp) - 1
hlm = high.add(low) / 2
ocm = openp.add(closep) / 2
ortamt = amount.where(pct > 0, (-1 * amount).where(pct < 0, 0))
ret30 = pd.read_pickle(dirct + 'zz500ret30min20200101_20200930').sort_index().query(
    "time in (1000,1030,1100,1130,1330,1400,1415,1430)")
# ret30rank = pd.read_pickle(dirct + 'zz500ret30minrank20200101_20200930').sort_index().query("time in (1000,1030,1100,1130,1330,1400,1430)")
# I = pd.DataFrame(1,index = ret30rank.index,columns = ret30rank.columns)

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(pd.DataFrame, 16), pd.DataFrame)


def replace0(df):
    return pd.DataFrame(bn.replace(df.values, 0, np.nan), index=df.index, columns=df.columns)


def add(left, right):
    return left.add(right)


pset.addPrimitive(add, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def mean2(df1, df2):
    return add(df1, df2) / 2


pset.addPrimitive(mean2, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def zmean2(df1, df2):
    return add(secz(df1), secz(df2)) / 2


pset.addPrimitive(zmean2, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def zmean3(df1, df2, df3):
    return (secz(df1).add(secz(df2)).add(secz(df3))) / 3


pset.addPrimitive(zmean3, [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)


def mean3(df1, df2, df3):
    return (df1.add(df2).add(df3)) / 3


pset.addPrimitive(mean3, [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)


def sub(left, right):
    return left.sub(right)


pset.addPrimitive(sub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def mul(left, right):
    return left.mul(right)


pset.addPrimitive(mul, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def div(left, right):
    return left.div(replace0(right))


pset.addPrimitive(div, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def neg(data):
    return data * -1


pset.addPrimitive(neg, [pd.DataFrame], pd.DataFrame)


def secz(df):
    return pd.DataFrame(
        (df.sub(bn.nanmean(df, axis=1), axis=0)).div(bn.replace(bn.nanstd(df, axis=1), 0, np.nan), axis=0),
        index=df.index, columns=df.columns)


pset.addPrimitive(secz, [pd.DataFrame], pd.DataFrame)


def zadd(df1, df2):
    return add(secz(df1), secz(df2))


pset.addPrimitive(zadd, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def zsub(df1, df2):
    return sub(secz(df1), secz(df2))


pset.addPrimitive(zsub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def secrank(df):
    return pd.DataFrame(bn.nanrankdata(df, axis=1), index=df.index, columns=df.columns)


pset.addPrimitive(secrank, [pd.DataFrame], pd.DataFrame)


def square(df):
    return df ** 2


pset.addPrimitive(square, [pd.DataFrame], pd.DataFrame)


def cube(df):
    return df ** 3


pset.addPrimitive(cube, [pd.DataFrame], pd.DataFrame)


def sqrt(df):
    return np.sqrt(abs(df))


pset.addPrimitive(sqrt, [pd.DataFrame], pd.DataFrame)


def logabsdif(df):
    logdata = np.log(abs(df))
    return logdata.sub(logdata.shift(1))


pset.addPrimitive(logabsdif, [pd.DataFrame], pd.DataFrame)


def logabsdifN(df, N):
    logdata = np.log(abs(df))
    return logdata.sub(logdata.shift(N))


pset.addPrimitive(logabsdifN, [pd.DataFrame, int], pd.DataFrame)


def log1abs(df):
    return np.log(abs(df + 1))


pset.addPrimitive(log1abs, [pd.DataFrame], pd.DataFrame)


def logabs(df):
    return np.log(pd.DataFrame(bn.replace(abs(df).values, 0, np.nan), index=df.index, columns=df.columns))


pset.addPrimitive(logabs, [pd.DataFrame], pd.DataFrame)


def logrank(df):
    return np.log(secrank(df))


pset.addPrimitive(logrank, [pd.DataFrame], pd.DataFrame)


# def rollParallel(df,n,func):
#    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group,n) for d,group in df.groupby('date'))
#    return pd.concat(retLst)

def rollsum(df, N):
    return pd.DataFrame(bn.move_sum(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollsum, [pd.DataFrame, int], pd.DataFrame)


# def gprollsum(df,N):
#    return df.groupby('date').apply(lambda x : rollsum(x,N))
# pset.addPrimitive(gprollsum,[pd.DataFrame,int],pd.DataFrame)

def rollabssum(df, N):
    return pd.DataFrame(bn.move_sum(abs(df), N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollabssum, [pd.DataFrame, int], pd.DataFrame)


# def gprollabssum(df,N):
#    return df.groupby('date').apply(lambda x : rollabssum(x,N))
# pset.addPrimitive(gprollabssum,[pd.DataFrame,int],pd.DataFrame)

def rollmean(df, N):
    return pd.DataFrame(bn.move_mean(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollmean, [pd.DataFrame, int], pd.DataFrame)


# def gprollmean(df,N):
#    return df.groupby('date').apply(lambda x : rollmean(x,N))
# pset.addPrimitive(gprollmean,[pd.DataFrame,int],pd.DataFrame)

def rollabsmean(df, N):
    return pd.DataFrame(bn.move_mean(abs(df), N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollabsmean, [pd.DataFrame, int], pd.DataFrame)


# def gprollabsmean(df,N):
#    return df.groupby('date').apply(lambda x : rollabsmean(x,N))
# pset.addPrimitive(gprollabsmean,[pd.DataFrame,int],pd.DataFrame)

def rollstd(df, N):
    return pd.DataFrame(bn.move_std(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollstd, [pd.DataFrame, int], pd.DataFrame)


# def gprollstd(df,N):
#    return df.groupby('date').apply(lambda x : rollstd(x,N))
# pset.addPrimitive(gprollstd,[pd.DataFrame,int],pd.DataFrame)

def rollskew(df, N):
    return df.rolling(N).skew()


pset.addPrimitive(rollskew, [pd.DataFrame, int], pd.DataFrame)


# def gprollskew(df,N):
#    return df.groupby('date').apply(lambda x : rollskew(x,N))
# pset.addPrimitive(gprollskew,[pd.DataFrame,int],pd.DataFrame)

def rollkurt(df, N):
    return df.rolling(N).kurt()


pset.addPrimitive(rollkurt, [pd.DataFrame, int], pd.DataFrame)


# def gprollkurt(df,N):
#    return df.groupby('date').apply(lambda x : rollkurt(x,N))
# pset.addPrimitive(gprollkurt,[pd.DataFrame,int],pd.DataFrame)

def rollmax(df, N):
    return pd.DataFrame(bn.move_max(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollmax, [pd.DataFrame, int], pd.DataFrame)


# def gprollmax(df,N):
#    return df.groupby('date').apply(lambda x : rollmax(x,N))
# pset.addPrimitive(gprollmax,[pd.DataFrame,int],pd.DataFrame)

def rollmin(df, N):
    return pd.DataFrame(bn.move_max(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollmin, [pd.DataFrame, int], pd.DataFrame)


# def gprollmin(df,N):
#    return df.groupby('date').apply(lambda x : rollmin(x,N))
# pset.addPrimitive(gprollmin,[pd.DataFrame,int],pd.DataFrame)

def rollvar(df, N):
    return pd.DataFrame(bn.move_var(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollvar, [pd.DataFrame, int], pd.DataFrame)


# def gprollvar(df,N):
#    return df.groupby('date').apply(lambda x : rollvar(x,N))
# pset.addPrimitive(gprollvar,[pd.DataFrame,int],pd.DataFrame)

def rollzscore(df, N):
    return pd.DataFrame(df.sub(bn.move_mean(df, N, axis=0)).div(bn.replace(bn.move_std(df, N, axis=0), 0, np.nan)),
                        index=df.index, columns=df.columns)


pset.addPrimitive(rollzscore, [pd.DataFrame, int], pd.DataFrame)


# def gprollz(df,N):
#    return df.groupby('date').apply(lambda x : rollzscore(x,N))
# pset.addPrimitive(gprollz,[pd.DataFrame,int],pd.DataFrame)

def rollrank(df, N):
    return pd.DataFrame(bn.move_rank(df, N, axis=0), index=df.index, columns=df.columns)


pset.addPrimitive(rollrank, [pd.DataFrame, int], pd.DataFrame)


# def gprollrank(df,N):
#    return df.groupby('date').apply(lambda x : rollrank(x,N))
# pset.addPrimitive(gprollrank,[pd.DataFrame,int],pd.DataFrame)

def lag(df, N):
    return df.shift(N)


pset.addPrimitive(lag, [pd.DataFrame, int], pd.DataFrame)


def sublag(df, N):
    return df.sub(df.shift(N))


pset.addPrimitive(sublag, [pd.DataFrame, int], pd.DataFrame)


def divlag(df, N):
    return df.div(replace0(df).shift(N))


pset.addPrimitive(divlag, [pd.DataFrame, int], pd.DataFrame)


def deltaN(df, N):
    return df.sub(lag(df, N))


pset.addPrimitive(deltaN, [pd.DataFrame, int], pd.DataFrame)


def grN(df, N):
    return df.div(replace0(lag(df, N)))


pset.addPrimitive(grN, [pd.DataFrame, int], pd.DataFrame)


def grsub1N(df, N):
    return grN(df, N) - 1


pset.addPrimitive(grsub1N, [pd.DataFrame, int], pd.DataFrame)


def hlccomp(df, N):
    return (df * 2).sub(rollmax(df, N).add(rollmin(df, N))).div(replace0(rollmax(df, N).sub(rollmin(df, N))))


pset.addPrimitive(hlccomp, [pd.DataFrame, int], pd.DataFrame)


def move2dist(df, N):
    return grsub1N(df, N).div(replace0(rollabssum(grsub1N(df, 1), N)))


pset.addPrimitive(move2dist, [pd.DataFrame, int], pd.DataFrame)


def move2std(df, N):
    return grsub1N(df, N).div(replace0(rollstd(grsub1N(df, 1), N)))


pset.addPrimitive(move2std, [pd.DataFrame, int], pd.DataFrame)


def move2hlvar(df, N):
    return grsub1N(df, N).div(replace0(rollmax(df, N).div(replace0(rollmin(df, N))) - 1))


pset.addPrimitive(move2hlvar, [pd.DataFrame, int], pd.DataFrame)


def move2hlp(df, N):
    return df.sub(df.shift(N)).div(replace0(rollmax(df, N + 1).sub(rollmin(df, N + 1))))


pset.addPrimitive(move2hlp, [pd.DataFrame, int], pd.DataFrame)


def hlvar(df, N):
    return rollmax(df, N).div(replace0(rollmin(df, N))) - 1


pset.addPrimitive(hlvar, [pd.DataFrame, int], pd.DataFrame)


def hlsub(df, N):
    return rollmax(df, N).sub(rollmin(df, N))


pset.addPrimitive(hlsub, [pd.DataFrame, int], pd.DataFrame)


def distinct(df, N):
    return rollabssum(grsub1N(df, 1), N)


pset.addPrimitive(distinct, [pd.DataFrame, int], pd.DataFrame)


def point2mean(df, N):
    return df.div(replace0(rollmean(df, N)))


pset.addPrimitive(point2mean, [pd.DataFrame, int], pd.DataFrame)


def point2sum(df, N):
    return df.div(replace0(rollsum(df, N)))


pset.addPrimitive(point2sum, [pd.DataFrame, int], pd.DataFrame)


def point2std(df, N):
    return df.div(replace0(rollstd(df, N)))


pset.addPrimitive(point2std, [pd.DataFrame, int], pd.DataFrame)


def pointsubmean(df, N):
    return df.sub(rollmean(df, N))


pset.addPrimitive(pointsubmean, [pd.DataFrame, int], pd.DataFrame)


def point2hlm(df, N):
    return (2 * df).div(replace0(rollmax(df, N).add(rollmin(df, N))))


pset.addPrimitive(point2hlm, [pd.DataFrame, int], pd.DataFrame)


def upshawpct(df, N):
    return rollmax(df, N + 1).sub(df.where(df >= df.shift(N), df.shift(N))).div(
        replace0(rollmax(df, N + 1).add(rollmin(df, N + 1))))


pset.addPrimitive(upshawpct, [pd.DataFrame, int], pd.DataFrame)


def dwnshawpct(df, N):
    return df.where(df <= df.shift(N), df.shift(N)).sub(rollmin(df, N + 1)).div(
        replace0(rollmax(df, N + 1).add(rollmin(df, N + 1))))


pset.addPrimitive(dwnshawpct, [pd.DataFrame, int], pd.DataFrame)


def pctsub(df, N1, N2):
    return df.div(replace0(df.shift(N1))).sub(df.div(replace0(df.shift(N2))))


pset.addPrimitive(pctsub, [pd.DataFrame, int, int], pd.DataFrame)


def accsub(df, N1, N2):
    return cvxp(df, N1).sub(cvxp(df, N2))


pset.addPrimitive(accsub, [pd.DataFrame, int, int], pd.DataFrame)


def maaccsub(df, N1, N2, N):
    return cvxp(rollmean(df, N1), N).sub(cvxp(rollmean(df, N2), N))


pset.addPrimitive(maaccsub, [pd.DataFrame, int, int, int], pd.DataFrame)


def pointpct2ma(df, N1, N):
    return df.div(replace0(df.shift(N))).sub(rollmean(df, N1).div(replace0(rollmean(df, N1).shift(N))))


pset.addPrimitive(pointpct2ma, [pd.DataFrame, int, int], pd.DataFrame)


def sigortchg(df1, df2):
    return df2.where(df1 > 0, -1 * df2)


pset.addPrimitive(sigortchg, [pd.DataFrame, pd.DataFrame], pd.DataFrame)


def rollsigmoid(df, N):
    return 1 / (1 + np.exp(neg(rollzscore(df, N))))


pset.addPrimitive(rollsigmoid, [pd.DataFrame, int], pd.DataFrame)


def seczsigmoid(df):
    return 1 / (1 + np.exp(neg(secz(df))))


pset.addPrimitive(seczsigmoid, [pd.DataFrame], pd.DataFrame)


def rsigmoidsin(df, N):
    return np.sin(rollsigmoid(df, N) * 2 * np.pi)


pset.addPrimitive(rsigmoidsin, [pd.DataFrame, int], pd.DataFrame)


def rsigmoidcos(df, N):
    return np.cos(rollsigmoid(df, N) * 2 * np.pi)


pset.addPrimitive(rsigmoidcos, [pd.DataFrame, int], pd.DataFrame)


def secsigmoidsin(df):
    return np.sin(seczsigmoid(df) * 2 * np.pi)


pset.addPrimitive(secsigmoidsin, [pd.DataFrame], pd.DataFrame)


def secsigmoidcos(df):
    return np.cos(seczsigmoid(df) * 2 * np.pi)


pset.addPrimitive(secsigmoidcos, [pd.DataFrame], pd.DataFrame)


# def corr(df1,df2,N):
#    part1 = (df1.mul(df2)).groupby('date').apply(lambda x: rollsum(x,N))
#    part2 = df1.groupby('date').apply(lambda x: rollmean(x,N))
#    part3 = df2.groupby('date').apply(lambda x: rollmean(x,N))
#    part4 = df1.groupby('date').apply(lambda x: rollstd(x,N))
#    part5 = df2.groupby('date').apply(lambda x: rollstd(x,N))
#    return ((part1/N).sub(part2.mul(part3))).div(replace0(part4.mul(part5)))
def corr(df1, df2, N):
    part1 = rollsum(df1.mul(df2), N)
    part2 = rollmean(df1, N)
    part3 = rollmean(df2, N)
    part4 = rollstd(df1, N)
    part5 = rollstd(df2, N)
    return ((part1 / N).sub(part2.mul(part3))).div(replace0(part4.mul(part5)))


pset.addPrimitive(corr, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)


# def cov(df1,df2,N):
#    part1 = (df1.mul(df2)).groupby('date').apply(lambda x: rollsum(x,N))
#    part2 = df1.groupby('date').apply(lambda x: rollmean(x,N))
#    part3 = df2.groupby('date').apply(lambda x: rollmean(x,N))
#    return (part1/N).sub(part2.mul(part3))

def cov(df1, df2, N):
    part1 = rollsum(df1.mul(df2), N)
    part2 = rollmean(df1, N)
    part3 = rollmean(df2, N)
    return (part1 / N).sub(part2.mul(part3))


pset.addPrimitive(cov, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)


# def corefunc(df):
#    return np.exp(df**2/-2)
# pset.addPrimitive(corefunc,[pd.DataFrame],pd.DataFrame)

def corefuncz(df):
    return np.exp(secz(df) ** 2 / -2)


pset.addPrimitive(corefuncz, [pd.DataFrame], pd.DataFrame)


# def revcorefunc(df):
#    return np.exp(df**2/2)
# pset.addPrimitive(revcorefunc,[pd.DataFrame],pd.DataFrame)

def revcorefuncz(df):
    return np.exp(secz(df) ** 2 / 2)


pset.addPrimitive(revcorefuncz, [pd.DataFrame], pd.DataFrame)


def ma1divma2(df, N1, N2):
    return rollmean(df, N1).div(replace0(rollmean(df, N2))) - 1


pset.addPrimitive(ma1divma2, [pd.DataFrame, int, int], pd.DataFrame)


def ma1subma2(df, N1, N2):
    return rollmean(df, N1).sub(rollmean(df, N2))


pset.addPrimitive(ma1subma2, [pd.DataFrame, int, int], pd.DataFrame)


def cvxp(df, N):
    return df.sub(df.shift(2 * N)).div(2 * replace0(df.shift(N)))


pset.addPrimitive(cvxp, [pd.DataFrame, int], pd.DataFrame)


def sub1(df):
    return df - 1


pset.addPrimitive(sub1, [pd.DataFrame], pd.DataFrame)


def sign(df):
    pos = pd.DataFrame(1, index=df.index, columns=df.columns)
    neg = pd.DataFrame(-1, index=df.index, columns=df.columns)
    zero = pd.DataFrame(0, index=df.index, columns=df.columns)
    rdf = pos.where(df > 0, neg)
    rdf = rdf.where(df != 0, zero)
    return rdf


# def cumrs(df,N):
#    return (df.where(df > 0).fillna(0).rolling(N).sum()).div(abs(df).rolling(N).sum())

def m2dlogabs(df, N):
    return move2dist(logabs(df), N)


pset.addPrimitive(m2dlogabs, [pd.DataFrame, int], pd.DataFrame)


def relu(df):
    return df.where(df > 0, 0)


pset.addPrimitive(relu, [pd.DataFrame], pd.DataFrame)


def prelu(df):
    return df.where(df > 0, df * 0.5)


pset.addPrimitive(prelu, [pd.DataFrame], pd.DataFrame)


def gtlastnumpct(df, N):
    return 0.5 - rollrank(df, N) / 2


pset.addPrimitive(gtlastnumpct, [pd.DataFrame, int], pd.DataFrame)


def smlastnumpct(df, N):
    return 0.5 + rollrank(df, N) / 2


pset.addPrimitive(smlastnumpct, [pd.DataFrame, int], pd.DataFrame)


def cdt(df, df1, df2):
    return df1.where(df > 0, df2)


pset.addPrimitive(cdt, [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)


def pctmoveperamt():
    return pct.div(replace0(amount))


pset.addPrimitive(pctmoveperamt, [], pd.DataFrame)


def pctmoveperamtNm(N):
    return (pct.div(replace0(amount))).rolling(N).mean()


pset.addPrimitive(pctmoveperamtNm, [int], pd.DataFrame)


def pctmoveperamtup():
    return (pct.div(replace0(amount))).where(pct > 0, 0)


pset.addPrimitive(pctmoveperamtup, [], pd.DataFrame)


def pctmoveperamtdwn():
    return (pct.div(replace0(amount))).where(pct < 0, 0)


pset.addPrimitive(pctmoveperamtdwn, [], pd.DataFrame)


def int_1():
    return 1


def int_2():
    return 2


def int_3():
    return 3


def int_4():
    return 4


def int_5():
    return 5


def int_7():
    return 7


def int_8():
    return 8


def int_9():
    return 9


def int_10():
    return 10


def int_11():
    return 11


def int_13():
    return 13


def int_14():
    return 14


def int_15():
    return 15


def int_16():
    return 16


def int_17():
    return 17


def int_20():
    return 20


def int_21():
    return 21


def int_24():
    return 24


def int_27():
    return 27


def int_60():
    return 60


def int_90():
    return 90


def int_120():
    return 120


pset.addPrimitive(int_1, [], int)
pset.addPrimitive(int_2, [], int)
pset.addPrimitive(int_3, [], int)
pset.addPrimitive(int_4, [], int)
pset.addPrimitive(int_5, [], int)
pset.addPrimitive(int_7, [], int)
pset.addPrimitive(int_8, [], int)
pset.addPrimitive(int_9, [], int)
pset.addPrimitive(int_10, [], int)
pset.addPrimitive(int_11, [], int)
pset.addPrimitive(int_13, [], int)
pset.addPrimitive(int_14, [], int)
pset.addPrimitive(int_15, [], int)
pset.addPrimitive(int_16, [], int)
pset.addPrimitive(int_17, [], int)
pset.addPrimitive(int_20, [], int)
pset.addPrimitive(int_21, [], int)
pset.addPrimitive(int_24, [], int)
pset.addPrimitive(int_60, [], int)
pset.addPrimitive(int_90, [], int)
pset.addPrimitive(int_120, [], int)
pset.addEphemeralConstant("const1", lambda: random.randint(1, 30), int)
pset.addEphemeralConstant("const2", lambda: random.randint(1, 30), int)
pset.addEphemeralConstant("const3", lambda: random.randint(1, 30), int)
pset.addEphemeralConstant("const4", lambda: random.randint(1, 30), int)

pset.renameArguments(ARG0='amount')
pset.renameArguments(ARG1='benchopen')
pset.renameArguments(ARG2='closep')
# pset.renameArguments(ARG3='cumhigh')
# pset.renameArguments(ARG4='cumlow')
# pset.renameArguments(ARG5='cumvol')
pset.renameArguments(ARG3='high')
pset.renameArguments(ARG4='low')
pset.renameArguments(ARG5='openp')
pset.renameArguments(ARG6='pct')
pset.renameArguments(ARG7='vol')
pset.renameArguments(ARG8='vwap')
pset.renameArguments(ARG9='hlp')
pset.renameArguments(ARG10='hlr')
pset.renameArguments(ARG11='ocp')
pset.renameArguments(ARG12='ocr')
pset.renameArguments(ARG13='hlm')
pset.renameArguments(ARG14='ocm')
pset.renameArguments(ARG15='ortamt')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


def evalSymbReg(individual, ret30, amount, benchopen, closep, high, low, openp, pct, vol, vwap, hlp, hlr, ocp, ocr, hlm,
                ocm, ortamt):
    function = toolbox.compile(expr=individual)
    res = function(amount, benchopen, closep, high, low, openp, pct, vol, vwap, hlp, hlr, ocp, ocr, hlm, ocm, ortamt)
    res1 = res.where(ret30.isnull() == False).dropna(axis=0, how='all')
    #    factor_rank = pd.DataFrame(secrank(res1),index = res1.index,columns = res1.columns)
    #    factor_rank_max = pd.Series(bn.nanmax(factor_rank,axis = 1),index = factor_rank.index)
    cor = res1.corrwith(ret30, axis=1).dropna()
    res_mean = cor.mean()
    res_std = cor.std()
    if (len(cor) < len(ret30) * 0.9) or (res_mean == 0) or (res_std == 0):
        return -np.inf,  # -np.inf,
    else:
        cot = res1.count()
        if len(cot.where(cot <= len(res1) * 0.9).dropna()) >= len(res1.columns) * 0.5:
            return -np.inf,
            '''
            if res_mean > 0:
                condition = factor_rank.sub(factor_rank_max*9/10,axis=0)
                return_condition = (ret30.where(condition>0))
                return_mean = return_condition.mean(axis=1).sub(ret30.mean(axis = 1))
                return_series = return_mean.dropna()

            else:
                condition = factor_rank.sub(factor_rank_max/10,axis=0)
                return_condition = (ret30.where(condition<0))
                return_mean = return_condition.mean(axis=1).sub(ret30.mean(axis = 1))
                return_series = return_mean.dropna()
            '''

        else:
            return abs(res_mean),  # abs(res_mean/res_std),


toolbox.register("evaluate", evalSymbReg, ret30=ret30, amount=amount, benchopen=benchopen, closep=closep, high=high,
                 low=low, openp=openp, pct=pct, vol=vol, vwap=vwap, hlp=hlp, hlr=hlr, ocp=ocp, ocr=ocr, hlm=hlm,
                 ocm=ocm, ortamt=ortamt)
toolbox.register("select1", tools.selBest)  # select数量
toolbox.register("select2", tools.selTournament, tournsize=4)  # select数量
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def main():
    pop = toolbox.population(n=60)
    #    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats用来汇报结果
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    #    mstats.register("avg", np.nanmean)
    #    mstats.register("std", np.nanstd)
    #    mstats.register("min", np.nanmin)
    mstats.register("max", np.nanmax)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 15, stats=mstats,  # halloffame=hof,
                                   verbose=True)
    # print log
    return pop, log  # , hof


if __name__ == '__main__':
    print(datetime.datetime.now())
    pool = Pool(24)

    toolbox.register("map", pool.map)
    print("start_programming")
    pop, log = main()
    #    pop,log,hof = main()
    '''
    top = tools.selBest(pop, k = 20)
    noDupes = []
    print(datetime.datetime.now())
    [noDupes.append(i) for i in top if not noDupes.count(i)]
    t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #输出
    num = len(noDupes)
    a = reporting('interday_' + t)

    for i in range(num):
        tree = gp.PrimitiveTree(noDupes[i])
        function = gp.compile(tree,pset)
        try:
            res = function(amount,benchopen,closep,high,low,openp,pct,vol,vwap,ortamt)
            #############################################################################  
            res1 = res.where(ret15rank.isnull()==False).dropna(axis=0,how='all')
            #######################################################################
            factor_rank = pd.DataFrame(secrank(res1),index = res1.index,columns = res1.columns)
            factor_rank_max = pd.Series(bn.nanmax(factor_rank,axis = 1),index = factor_rank.index)

            cor = factor_rank.corrwith(ret15rank,axis=1).sort_index()
            res_mean = cor.mean()
            res_std = cor.std()
            if res_mean > 0:
                condition = factor_rank.sub(factor_rank_max*9/10,axis=0)
                return_condition = (ret15.where(condition>0))
                return_mean = return_condition.mean(axis=1).sub(ret15.mean(axis=1))
                return_series = return_mean.dropna()
            else:
                condition = factor_rank.sub(factor_rank_max/10,axis=0)
                return_condition = (ret15.where(condition<0))
                return_mean = return_condition.mean(axis=1).sub(ret15.mean(axis=1))
                return_series = return_mean.dropna() 
            a.addSectionName(str(i) + "." +"expression"+"\n"+str(tree))  
            a.addParagraph("IC value:" + str(res_mean) + "\n" +"IR_value:" + str(res_mean/res_std))
            a.addDataPlot(cor.cumsum())
            a.addParagraph("top IR_value:" + str(return_series.mean()/return_series.std()))
            a.addDataPlot(return_series.cumsum())
            print(str(tree),res_mean,res_mean/res_std)
        except:
            continue        
    a.turnDocx()   
    '''


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:57:01 2022

@author: user
"""
import datetime
import pandas as pd
import numpy as np
import bottleneck as bn
import random
import gc
from numba import jit
# import talib
import warnings

warnings.filterwarnings('ignore')


# 运算函数


def abs(df):
    return np.abs(df)


def abs_ln(df):
    return np.log(abs(df))


def abs_ln_add1(df):
    return np.log(np.abs(df + 1))


def neg(df):
    return -1 * df


def sub1(df):
    return df - 1


def square(df):
    return np.power(df, 2)


def cube(df):
    return np.power(df, 3)


def abssqrt(df):
    return np.sqrt(np.abs(df))


def log1abs(df):
    return np.log(abs(df) + 1)


def tanh(df):
    return np.tanh(df)


def sign(df):
    return np.sign(df)


def ma(df, N):
    return bn.move_mean(df, window=N, axis=0)


def absma(df, N):
    return bn.move_mean(abs(df), window=N, axis=0)


def double_ma_diff(df, N):
    m = ma(df, N)
    return m - ma(m, N)


def shift(df, N):
    m, n = np.shape(df)
    return np.concatenate([np.array([[np.nan for i in range(n)] for j in range(N)]), df[:-N]])


def power(df, N):
    return np.power(df, N)


def ema(df, alpha):
    m, n = np.shape(df)
    df = np.where(np.isnan(df), 0, df)
    data = np.zeros((m, n))
    for i in range(m):
        data[i] = df[i] if i == 0 else alpha * df[i] + (1 - alpha) * data[i - 1]
    return data


def rollstd(df, N):
    return bn.move_std(df, window=N, axis=0)


def rollmax(df, N):
    return bn.move_max(df, window=N, axis=0)


def rollmin(df, N):
    return bn.move_min(df, window=N, axis=0)


def rollmedian(df, N):
    return bn.move_median(df, window=N, axis=0)


def rollrange(df, N):
    return rollmax(df, N) - rollmin(df, N)


def rolldivrange(df, N):
    return div(rollmax(df, N), rollmin(df, N))


def rolldivrgratio(df, N):
    return div(rollmax(df, N), rollmin(df, N)) - 1


def submean(df, N):
    return df - ma(df, N)


def submeansft1(df, N):
    return df - shift(ma(df, N), 1)


def divmean(df, N):
    return div(df, ma(df, N))


def divmeansft1(df, N):
    return div(df, shift(ma(df, N), 1))


def divrollstd(df, N):
    return div(df, rollstd(df, N))


def divrollstdsft1(df, N):
    return div(df, shift(rollstd(df, N), 1))


def sharp(df, N):
    return div(submean(df, N), rollstd(df, N))


def sharpsft1(df, N):
    return div(df - shift(ma(df, N), 1), shift(rollstd(df, N), 1))


def m2absm(df, N):
    return div(ma(df, N), ma(abs(df), N))


def m2m(df, N1, N2):
    return div(ma(df, N1), ma(df, N2))


def m2std(df, N):
    return div(ma(df, N), rollstd(df, N))


def rollmapct(df, N, N1):
    m = ma(df, N)
    return grlag(m, N1)


def atr(df, N):
    return np.maximum(rollmax(df, N) - rollmin(df, N), rollmax(df, N) - shift(df, N), shift(df, N) - rollmin(df, N))


def atrpct(df, N):
    return np.maximum(div(rollmax(df, N), rollmin(df, N)) - 1, div(rollmax(df, N), shift(df, N)) - 1,
                      div(shift(df, N), rollmin(df, N)) - 1)


def rollatr(df, N1, N2):
    return ma(atr(df, N1), N2)


def rollatrpct(df, N1, N2):
    return ma(atrpct(df, N1), N2)


def abssubmean(df, N):
    return abs(df - ma(df, N))


def rollskew(df, N):
    return pd.DataFrame(df).rolling(N).skew().values


def rollkurt(df, N):
    return pd.DataFrame(df).rolling(N).kurt().values


def corr2(left, right, N):
    xy = bn.move_mean(left * right, window=N, axis=0)
    x = bn.move_mean(left, window=N, axis=0)
    y = bn.move_mean(right, window=N, axis=0)
    x2 = bn.move_mean(power(left, 2), window=N, axis=0)
    y2 = bn.move_mean(power(right, 2), window=N, axis=0)
    up = xy - x * y
    down = np.sqrt((x2 - power(x, 2)) * (y2 - power(y, 2)))
    return div(up, down)


def selfcorr(df, N, N1):
    return corr2(df, shift(df, N1), N)


def countupdown(df, N):
    return bn.move_mean(sign(df), window=N, axis=0)


def add(df1, df2):
    return df1 + df2


def sub(df1, df2):
    return df1 - df2


def mul(df1, df2):
    return df1 * df2


def div(df1, df2):
    return df1 / np.where(df2 == 0, np.nan, df2)


def divselflag(df, N):
    return div(df, shift(df, N))


def grlag(df, N):
    return div(df, shift(df, N)) - 1


def selftsimb(df, N):
    return div(sub(df, shift(df, N)), abs(df) + abs(shift(df, N)))


def divsft1(df1, df2):
    return df1.div(shift(df2, 1))


def divsub1(df1, df2):
    return sub1(div(df1, df2))


def pos_data(df):
    return np.where(df > 0, df, 0)


def neg_data(df):
    return np.where(df < 0, df, 0)


def posnegcount_gapratio(df, N):
    return div(ma(pos_data(sign(df)), N) + (ma(neg_data(sign(df)), N)),
               ma(pos_data(sign(df)), N) - (ma(neg_data(sign(df)), N)))


def sub2add(df1, df2):
    return div(sub(df1, df2), add(df1, df2))


def distance(df, N):
    return absma(grlag(df, 1), N)


def p2hlm(df, N):
    return div(2 * df, rollmax(df, N) + rollmin(df, N)) - 1


def upshawpct(df, N):
    return div(rollmax(df, N + 1) - np.where(df >= shift(df, N), df, shift(df, N)),
               rollmax(df, N + 1) + rollmin(df, N + 1))


def dwnshawpct(df, N):
    return div(np.where(df <= shift(df, N), df, shift(df, N)) - rollmin(df, N + 1),
               rollmax(df, N + 1) + rollmin(df, N + 1))


def updwnsdwgap(df, N):
    return upshawpct(df, N) - dwnshawpct(df, N)


def sigmoid(df):
    return 1 / (1 + np.exp(neg(df)))


def rollsigmoid(df, N):
    return 1 / (1 + np.exp(neg(sharp(df, N))))


def logret(df, N):
    return abs_ln(div(df, shift(df, N)))


def divVol(df, Vol):
    return div(df, Vol)


def divVolsqrt(df, Vol):
    return div(df, abssqrt(Vol))


def corr2Vol(df, N, Vol):
    return corr2(df, Vol, N)


def corr2Volsqrt(df, N, Vol):
    return corr2(df, abssqrt(Vol), N)


def corr2OI(df, N,OI):
    return corr2(df, OI, N)


def corr2OIsqrt(df, N):
    return corr2(df, sqsrtOIsqrt, N)




def corr2OIchg(df, N):
    return corr2(df, OIchg, N)



def corr2OIchgr(df, N):
    return corr2(df, OIchgr, N)



def corr2Volchg(df, N):
    return corr2(df, Volchg, N)



def corr2Volchgr(df, N):
    return corr2(df, Volchgr, N)



def corr2Vol2OI(df, N):
    return corr2(df, Vol2OI, N)



def corr2Volroll(df, N, N1):
    return corr2(df, ma(Vol, N1), N)



def corr2Volsqrtroll(df, N, N1):
    return corr2(df, ma(Volsqrt, N1), N)



def corr2OIroll(df, N, N1):
    return corr2(df, ma(OI, N1), N)



def corr2OIsqrtroll(df, N, N1):
    return corr2(df, ma(OIsqrt, N1), N)



def corr2OIchgroll(df, N, N1):
    return corr2(df, ma(OIchg, N1), N)



def corr2OIchgrroll(df, N, N1):
    return corr2(df, ma(OIchgr, N1), N)




def corr2Volchgroll(df, N, N1):
    return corr2(df, ma(Volchg, N1), N)




def corr2Volchgrroll(df, N, N1):
    return corr2(df, ma(Volchgr, N1), N)



def corr2Vol2OIroll(df, N, N1):
    return corr2(df, ma(Vol2OI, N1), N)



def Volweight(df, N):
    return div(mul(df, Vol).rolling(N).mean(), Vol.rolling(N).mean())




def Volsqrtweight(df, N):
    return div(mul(df, Volsqrt).rolling(N).mean(), Volsqrt.rolling(N).mean())




def Vol2OIweight(df, N):
    return div(mul(df, Vol2OI).rolling(N).mean(), Vol2OI.rolling(N).mean())



def KAMA(df, N):
    return np.apply_along_axis(talib.KAMA, 0, df, N)




def CMO(df, N):
    return np.apply_along_axis(talib.CMO, 0, df, N) / 100



def MACD(df, N1, N2, N3):
    return np.apply_along_axis(talib.MACD, 0, df, N1, N2, N3)[0]



def DX(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.DX(High.T[i], Low.T[i], Close.T[i], timeperiod=N)))
    return np.array(temp).T / 100



# ATR(high, low, close, timeperiod=14)
def ATR(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.ATR(High.T[i], Low.T[i], Close.T[i], timeperiod=N)))
    return np.array(temp).T



def NATR(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.NATR(High.T[i], Low.T[i], Close.T[i], timeperiod=N)))
    return np.array(temp).T



# TRANGE(high, low, close)
def TRANGE():
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.TRANGE(High.T[i], Low.T[i], Close.T[i])))
    return np.array(temp).T



# AD(high, low, close, volume)
def AD():
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.AD(High.T[i], Low.T[i], Close.T[i], Vol.T[i])))
    return np.array(temp).T



# ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
def ADOSC(N1, N2):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.ADOSC(High.T[i], Low.T[i], Close.T[i], Vol.T[i], N1, N2)))
    return np.array(temp).T



# OBV(close, volume)
def OBV():
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.OBV(Close.T[i], Vol.T[i])))
    return np.array(temp).T



# ADX(high, low, close, timeperiod=14)
def ADX(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.ADX(High.T[i], Low.T[i], Close.T[i], N)))
    return np.array(temp).T / 100



# ADXR(high, low, close, timeperiod=14)
def ADXR(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.ADXR(High.T[i], Low.T[i], Close.T[i], N)))
    return np.array(temp).T / 100




# APO(close, fastperiod=12, slowperiod=26, matype=0)
def APO(df, N1, N2):
    return np.apply_along_axis(talib.APO, 0, df, N1, N2)




# real = AROONOSC(high, low, timeperiod=14)
def AROONOSC(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.AROONOSC(High.T[i], Low.T[i], N)))
    return np.array(temp).T / 100



# real = BOP(open, high, low, close)
def BOP():
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.BOP(Open.T[i], High.T[i], Low.T[i], Close.T[i])))
    return np.array(temp).T



# CCI(high, low, close, timeperiod=14)
def CCI(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.CCI(High.T[i], Low.T[i], Close.T[i], N)))
    return np.array(temp).T * 0.015




# MFI(high, low, close, volume, timeperiod=14)
def MFI(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.MFI(High.T[i], Low.T[i], Close.T[i], Vol.T[i], N)))
    return np.array(temp).T / 100




def MOM(df, N):
    return np.apply_along_axis(talib.MOM, 0, df, N)




# real = PPO(close, fastperiod=12, slowperiod=26, matype=0)
def PPO(df, N1, N2):
    return np.apply_along_axis(talib.PPO, 0, df, N1, N2, 0)



# real = ROC(close, timeperiod=10)
def ROC(df, N):
    return np.apply_along_axis(talib.ROC, 0, df, N)



def ROCP(df, N):
    return np.apply_along_axis(talib.ROCP, 0, df, N)



def ROCR(df, N):
    return np.apply_along_axis(talib.ROCR, 0, df, N)




# RSI(close, timeperiod=14)
def RSI(df, N):
    return np.apply_along_axis(talib.RSI, 0, df, N) / 100




# TRIX(close, timeperiod=30)
def TRIX(df, N):
    return np.apply_along_axis(talib.TRIX, 0, df, N)



# ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
def ULTOSC(N1, N2, N3):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.ULTOSC(High.T[i], Low.T[i], Close.T[i], N1, N2, N3)))
    return np.array(temp).T



# WILLR(high, low, close, timeperiod=14)
def WILLR(N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.WILLR(High.T[i], Low.T[i], Close.T[i], N)))
    return np.array(temp).T / -100




# BETA(high, low, timeperiod=5)
def BETA(df1, df2, N):
    temp = []
    for i in range(len(col)):
        temp.append(list(talib.BETA(df1.T[i], df2.T[i], N)))
    return np.array(temp).T




# real = LINEARREG(close, timeperiod=14)
def LINEARREG_SLOPE(df, N):
    return np.apply_along_axis(talib.LINEARREG_SLOPE, 0, df, N)



# LINEARREG_ANGLE(close, timeperiod=14)
def LINEARREG_ANGLE(df, N):
    return np.apply_along_axis(talib.LINEARREG_ANGLE, 0, df, N) / 90




# 参数
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


def cummax(df):
    m = len(df)
    data = np.zeros(len(df))
    maxv = 0
    for i in range(m):
        if np.isnan(df[i]):
            continue
        else:
            maxv = max(maxv, df[i])
            data[i] = maxv
    return data


def cummaximum(df):
    m, n = np.shape(df)
    data = np.zeros((m, n))
    maxv = df[0,]
    for i in range(m):
        maxv = np.maximum(maxv, df[i])
        data[i] = maxv
    return data



def preprocss(res):
    res1 = np.where(res == -np.inf, np.nan, res)
    res1 = np.where(res1 == np.inf, np.nan, res1)
    res1 = res1 / bn.nanstd(res1, axis=0)
    # 合并等权净值
    nvret = np.tanh(res1) * ret
    secsum = np.sum(nvret, axis=1)
    return nvret, secsum

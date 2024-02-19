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

warnings.filterwarnings('ignore')
import sys

sys.path.append('d:\\factor_minning')
from pypdf import reporting

import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFutBasic

tickers_list = ConstFutBasic.fut_code_list
start_date = '2019-01-01'
end_date = '2022-08-31'

close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq='D', index=1, ret_index=False)
open = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq='D', index=1, ret_index=False)
high = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='high', freq='D', index=1, ret_index=False)
low = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='low',
                                        freq='D', index=1, ret_index=False)
volume = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='vol', freq='D', index=1, ret_index=False)

oi = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='oi',
                                       freq='D', index=1, ret_index=False)

begin_date = '20200102'
end_date = '20220826'
cursor, conn = initDB()
cursor.execute(
    f"select code,date,open_normal,close_normal,high_normal,low_normal,open_rear,close_rear,high_rear,low_rear,vwap,adjfactor,istrade,abnormal from dailytrade where ipo1Y > 0 and date >= '{begin_date}' and date <= '{end_date}'")
data = pd.DataFrame(cursor.fetchall())

close_n = pd.pivot_table(data, index='date', columns='code', values='close_normal').sort_index()
close_r = pd.pivot_table(data, index='date', columns='code', values='close_rear').sort_index()

open_n = pd.pivot_table(data, index='date', columns='code', values='open_normal').sort_index()
open_r = pd.pivot_table(data, index='date', columns='code', values='open_rear').sort_index()

high_n = pd.pivot_table(data, index='date', columns='code', values='high_normal').sort_index()
high_r = pd.pivot_table(data, index='date', columns='code', values='high_rear').sort_index()

low_n = pd.pivot_table(data, index='date', columns='code', values='low_normal').sort_index()
low_r = pd.pivot_table(data, index='date', columns='code', values='low_rear').sort_index()

vwap_n = pd.pivot_table(data, index='date', columns='code', values='vwap').sort_index()
adjfactor = pd.pivot_table(data, index='date', columns='code', values='adjfactor').sort_index()
vwap_r = vwap_n.mul(adjfactor)

istrade = pd.pivot_table(data, index='date', columns='code', values='istrade').sort_index()
abnormal = pd.pivot_table(data, index='date', columns='code', values='abnormal').sort_index()
I = pd.DataFrame(1, columns=istrade.columns, index=istrade.index)
# ret = (open_r.shift(-1).div(open_r) - 1)
# cursor.execute(f"select date,openprice from indexpct where date >= '{begin_date}' and date <= '{end_date}' and code = '000905.SH'")
# indexdata = pd.DataFrame(cursor.fetchall()).set_index('date').sort_index()
# indexret = (indexdata.shift(-1).div(indexdata) - 1)['openprice']
# excessret = ret#.subtract(indexret,axis = 0)
# excessret = excessret.where((abnormal == 0)&(istrade > 0)).shift(-1)

cursor.execute(
    f"select code,date,IDVP12,IDVP20 from factor_value where date >= '{begin_date}' and date <= '{end_date}'")
factor = pd.DataFrame(cursor.fetchall())

# IDVP12 = pd.pivot_table(factor,index = 'date',columns = 'code',values = 'IDVP12').sort_index()
IDVP20 = pd.pivot_table(factor, index='date', columns='code', values='IDVP20').sort_index()

conn = pymssql.connect(host='DESKTOP-JTBF9HN', user='sa', password='Alpha123', database='HF_Data', charset='GBK')
cursor = conn.cursor(as_dict=True)
cursor.execute(
    f"select code,date,low from trade_30min where time = '1000' and date >= '{begin_date}' and date <= '{end_date}'")
firsthalf = pd.DataFrame(cursor.fetchall())
fhlow = pd.pivot_table(firsthalf, index='date', columns='code', values='low').sort_index()
fhlow_r = adjfactor.mul(fhlow)[istrade.columns.to_list()]

cursor.execute(
    f"select code,date,closeprice from trade_30min where time = '1130' and date >= '{begin_date}' and date <= '{end_date}'")
secondhalf = pd.DataFrame(cursor.fetchall())
shclose = pd.pivot_table(secondhalf, index='date', columns='code', values='closeprice').sort_index()
shclose_r = adjfactor.mul(shclose)[istrade.columns.to_list()]

retnormal = (shclose_r.shift(-1) * (1 - 0.000125)).div(open_r * (1 + 0.00025)) - 1
rethighopen = (shclose_r.shift(-1) * (1 - 0.000125)).div(close_r.shift(1) * (1 + 0.00025) * 1.01) - 1

ret = (retnormal.where(open_r <= close_r.shift(1), rethighopen.where(fhlow_r <= close_r.shift(1), 0))).shift(-1)

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
    ma = close_r.rolling(N).mean()
    sig = I.where(close_r > ma, 0)
    return sig


pset.addPrimitive(closeoverma, [int], pd.DataFrame)


# 收于均线上方
def closebelowma(N):
    ma = close_r.rolling(N).mean()
    sig = I.where(close_r < ma, 0)
    return sig


pset.addPrimitive(closebelowma, [int], pd.DataFrame)


# 均线上穿
def ma1overma2(N1, N2):
    ma1 = close_r.rolling(N1).mean()
    ma2 = close_r.rolling(N2).mean()
    sig = I.where(ma1 > ma2, 0)
    return sig


pset.addPrimitive(ma1overma2, [int, int], pd.DataFrame)


# 长上影线
def longupshadow(pct):
    maxop = open_r.where(open_r > close_r, close_r)
    upshadow = high_r.sub(maxop)
    sig = I.where(upshadow.div(high_r.sub(low_r)) > pct, 0)
    return sig


pset.addPrimitive(longupshadow, [float], pd.DataFrame)


# N天K长上影线
def longupshadowN(pct, N):
    openn = open_r.shift(N)
    highn = high_r.rolling(N).max()
    lown = low_r.rolling(N).min()
    maxop = openn.where(openn > close_r, close_r)
    upshadow = highn.sub(maxop)
    sig = I.where(upshadow.div(highn.sub(lown)) > pct, 0)
    return sig


pset.addPrimitive(longupshadowN, [float, int], pd.DataFrame)


# 长下影线
def longdownshadow(pct):
    minop = open_r.where(open_r < close_r, close_r)
    downshadow = minop.sub(low_r)
    sig = I.where(downshadow.div(high_r.sub(low_r)) > pct, 0)
    return sig


pset.addPrimitive(longdownshadow, [int], pd.DataFrame)


# N天长下影线
def longdownshadowN(pct, N):
    openn = open_r.shift(N)
    highn = high_r.rolling(N).max()
    lown = low_r.rolling(N).min()
    minop = openn.where(openn < close_r, close_r)
    downshadow = minop.sub(lown)
    sig = I.where(downshadow.div(highn.sub(lown)) > pct, 0)
    return sig


pset.addPrimitive(longdownshadowN, [float, int], pd.DataFrame)


# 短上影线
def shortupshadow(pct):
    maxop = open_r.where(open_r > close_r, close_r)
    upshadow = high_r.sub(maxop)
    sig = I.where(upshadow.div(high_r.sub(low_r)) < pct, 0)
    return sig


pset.addPrimitive(shortupshadow, [float], pd.DataFrame)


# N天短上影线
def shortupshadowN(pct, N):
    openn = open_r.shift(N)
    highn = high_r.rolling(N).max()
    lown = low_r.rolling(N).min()
    maxop = openn.where(openn > close_r, close_r)
    upshadow = highn.sub(maxop)
    sig = I.where(upshadow.div(highn.sub(lown)) < pct, 0)
    return sig


pset.addPrimitive(shortupshadowN, [float, int], pd.DataFrame)


# 短下影线
def shortdownshadow(pct):
    minop = open_r.where(open_r < close_r, close_r)
    downshadow = minop.sub(low_r)
    sig = I.where(downshadow.div(high_r.sub(low_r)) < pct, 0)
    return sig


pset.addPrimitive(shortdownshadow, [float], pd.DataFrame)


# N天短下影线
def shortdownshadowN(pct, N):
    openn = open_r.shift(N)
    highn = high_r.rolling(N).max()
    lown = low_r.rolling(N).min()
    minop = openn.where(openn < close_r, close_r)
    downshadow = minop.sub(lown)
    sig = I.where(downshadow.div(highn.sub(lown)) < pct, 0)
    return sig


pset.addPrimitive(shortdownshadowN, [float, int], pd.DataFrame)


# 十字星
def crossstar(pct):
    tang = abs(open_r.sub(close_r))
    alllen = high_r.sub(low_r)
    sig = I.where(tang.div(alllen) < pct, 0)
    return sig


pset.addPrimitive(crossstar, [float], pd.DataFrame)


# N天十字星
def crossstarN(pct, N):
    openn = open_r.shift(N)
    highn = high_r.rolling(N).max()
    lown = low_r.rolling(N).min()
    tang = abs(openn.sub(close_r))
    alllen = highn.sub(lown)
    sig = I.where(tang.div(alllen) < pct, 0)
    return sig


pset.addPrimitive(crossstarN, [float, int], pd.DataFrame)


# 盘中突破N天新高
def highnewhigh(N):
    sig = I.where(high_r >= high_r.rolling(N).max(), 0)
    return sig


pset.addPrimitive(highnewhigh, [int], pd.DataFrame)


# 收盘突破N天新高
def closenewhigh(N):
    sig = I.where(close_r >= close_r.rolling(N).max(), 0)
    return sig


pset.addPrimitive(closenewhigh, [int], pd.DataFrame)


# 盘中突破N天新低
def lownewlow(N):
    sig = I.where(low_r <= low_r.rolling(N).min(), 0)
    return sig


pset.addPrimitive(lownewlow, [int], pd.DataFrame)


# 收盘突破N天新低
def closenewlow(N):
    sig = I.where(close_r <= close_r.rolling(N).min(), 0)
    return sig


pset.addPrimitive(closenewlow, [int], pd.DataFrame)


# 开盘跳空低开低于N天最低
def openjumplowN(N):
    histlow = low_r.rolling(N).min().shift(1)
    sig = I.where(open_r < histlow, 0)
    return sig


pset.addPrimitive(openjumplowN, [int], pd.DataFrame)


# 开盘跳空低开低于前一天天最低
def openjumplow():
    histlow = low_r.shift(1)
    sig = I.where(open_r < histlow, 0)
    return sig


pset.addPrimitive(openjumplow, [], pd.DataFrame)


# 收盘调控低开低于N天最低
def closejumplowN(N):
    histlow = low_r.rolling(N).min().shift(1)
    sig = I.where(close_r < histlow, 0)
    return sig


pset.addPrimitive(closejumplowN, [int], pd.DataFrame)


# 收盘调控低开低于前一天最低
def closejumplow():
    histlow = low_r.shift(1)
    sig = I.where(close_r < histlow, 0)
    return sig


pset.addPrimitive(closejumplow, [], pd.DataFrame)


# 开盘跳空高开
def openjumphighN(N):
    histhigh = high_r.rolling(N).max().shift(1)
    sig = I.where(open_r > histhigh, 0)
    return sig


pset.addPrimitive(openjumphighN, [int], pd.DataFrame)


# 开盘跳空高开
def openjumphigh():
    histhigh = high_r.shift(1)
    sig = I.where(open_r > histhigh, 0)
    return sig


pset.addPrimitive(openjumphigh, [], pd.DataFrame)


# 收盘跳空高开
def closejumphighN(N):
    histhigh = high_r.rolling(N).max().shift(1)
    sig = I.where(close_r > histhigh, 0)
    return sig


pset.addPrimitive(closejumphighN, [int], pd.DataFrame)


# 收盘跳空高开
def closejumphigh():
    histhigh = high_r.shift(1)
    sig = I.where(close_r > histhigh, 0)
    return sig


pset.addPrimitive(closejumphigh, [], pd.DataFrame)


# 开盘价高开
def openhigher():
    return I.where(open_r > close_r.shift(1), 0)


pset.addPrimitive(openhigher, [], pd.DataFrame)


# 开盘价低开
def openlower():
    return I.where(open_r < close_r.shift(1), 0)


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
    Nmean = abs(close_r.div(open_r) - 1).rolling(N).mean().shift(1)
    return I.where(abs(close_r.div(open_r) - 1) > Nmean, 0)


pset.addPrimitive(ocgrtNmean, [int], pd.DataFrame)


# 实体波动大于过去N天最大值
def ocgrtNmax(N):
    Nmax = abs(close_r.div(open_r) - 1).rolling(N).max().shift(1)
    return I.where(abs(close_r.div(open_r) - 1) > Nmax, 0)


pset.addPrimitive(ocgrtNmax, [int], pd.DataFrame)


# 实体波动小于过去N天最小值
def oclsNmin(N):
    Nmin = abs(close_r.div(open_r) - 1).rolling(N).min().shift(1)
    return I.where(abs(close_r.div(open_r) - 1) < Nmin, 0)


pset.addPrimitive(oclsNmin, [int], pd.DataFrame)


# 影线波动大于过去N天均值
def shgrtNmean(N):
    Nmean = abs(high_r.div(low_r) - 1).rolling(N).mean().shift(1)
    return I.where(abs(high_r.div(low_r) - 1) > Nmean, 0)


pset.addPrimitive(shgrtNmean, [int], pd.DataFrame)


# 影线波动大于过去N天最大值
def shgrtNmax(N):
    Nmax = abs(high_r.div(low_r) - 1).rolling(N).max().shift(1)
    return I.where(abs(high_r.div(low_r) - 1) > Nmax, 0)


pset.addPrimitive(shgrtNmax, [int], pd.DataFrame)


# 影线波动小于过去N天最小值
def shlsNmin(N):
    Nmin = abs(high_r.div(low_r) - 1).rolling(N).min().shift(1)
    return I.where(abs(high_r.div(low_r) - 1) < Nmin, 0)


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
    return I.where(close_r <= low_r, 0)


pset.addPrimitive(closelowest, [], pd.DataFrame)


# 收盘是最高价
def closehighest():
    return I.where(close_r >= high_r, 0)


pset.addPrimitive(closehighest, [], pd.DataFrame)


# 开盘是最低价
def openlowest():
    return I.where(open_r <= low_r, 0)


pset.addPrimitive(openlowest, [], pd.DataFrame)


# 开盘是最高价
def openhighest():
    return I.where(open_r >= high_r, 0)


pset.addPrimitive(openhighest, [], pd.DataFrame)


# 一字板 or 停牌？
def highlowsame():
    return I.where(high_r == low_r, 0)


pset.addPrimitive(highlowsame, [], pd.DataFrame)


# 当日上涨
def priceup():
    return I.where(close_r > close_r.shift(1), 0)


pset.addPrimitive(priceup, [], pd.DataFrame)


# N天上涨
def Ndayup(N):
    return I.where(close_r > close_r.shift(N), 0)


pset.addPrimitive(Ndayup, [int], pd.DataFrame)


# 红K线
def redK():
    return I.where(close_r > open_r, 0)


pset.addPrimitive(redK, [], pd.DataFrame)


# N天K线红
def redKN(N):
    return I.where(close_r > open_r.shift(N), 0)


pset.addPrimitive(redKN, [int], pd.DataFrame)


# 绿K线
def greenK():
    return I.where(close_r < open_r, 0)


pset.addPrimitive(greenK, [], pd.DataFrame)


# N天K线绿
def greenKN(N):
    return I.where(close_r < open_r.shift(N), 0)


pset.addPrimitive(greenKN, [int], pd.DataFrame)


# N天位移路程比超过阈值
def closemovedistratio(pct, N):
    move = abs(close_r.div(close_r.shift(N)) - 1)
    distance = abs(close_r.div(close_r.shift(1)) - 1).rolling(N).sum()
    return I.where(move.div(distance) > pct, 0)


pset.addPrimitive(closemovedistratio, [float, int], pd.DataFrame)


# K线实体相对前一日长
def Klonger():
    Klen = abs(close_r.div(open_r) - 1)
    return I.where(Klen > Klen.shift(1), 0)


pset.addPrimitive(Klonger, [], pd.DataFrame)


# K线实体相对前一日短
def Kshorter():
    Klen = abs(close_r.div(open_r) - 1)
    return I.where(Klen < Klen.shift(1), 0)


pset.addPrimitive(Kshorter, [], pd.DataFrame)


# N天K线实体相对前一期长
def NKlonger(N):
    NKlen = abs(close_r.div(open_r.shift(N)) - 1)
    return I.where(NKlen > NKlen.shift(N + 1), 0)


pset.addPrimitive(NKlonger, [int], pd.DataFrame)


# N天K线实体相对前一期短
def NKshorter(N):
    NKlen = abs(close_r.div(open_r.shift(N)) - 1)
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


def evalSymbReg(individual, ret, IDVP20, I):
    function = toolbox.compile(expr=individual)
    res = function(I)
    res1 = res.where(ret.isnull() == False).dropna(axis=0, how='all')
    #    idvp12rank = (-1*IDVP12).where(res1 > 0).rank(axis = 1)
    idvp20rank = (-1 * IDVP20).where(res1 > 0).rank(axis=1)
    #    totalrank = (idvp12rank.add(idvp20rank)).rank(axis = 1)

    ret_cond = ret.where(idvp20rank <= 15)
    ret_mean = ret_cond.sum(axis=1) / 15
    if (ret_mean.isnull().sum() > len(ret_cond) * 0.8) or (len(ret_cond) == 0) or (ret_mean.std() == 0):
        return -np.inf, -np.inf,
    else:
        return ret_mean.dropna().mean() / ret_mean.dropna().std(), ret_mean.fillna(0).sum(),


toolbox.register("evaluate", evalSymbReg, ret=ret, IDVP20=IDVP20, I=I)
# toolbox.register("select1", tools.selBest)#select数量
toolbox.register("select", tools.selTournament, tournsize=4)  # select数量
##两棵树每棵树各选一颗，然后交换其中一个
toolbox.register("mate", gp.cxOnePoint)
##产生一个表达式，所有的叶子结点有相同的长度
toolbox.register("expr_mut", gp.genFull, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
##不能交配过长
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))


def main():
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats用来汇报结果
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.nanmean)
    mstats.register("std", np.nanstd)
    mstats.register("min", np.nanmin)
    mstats.register("max", np.nanmax)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 8, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ != "__main__":
    pool = Pool(10)
    toolbox.register("map", pool.map)
    print("start_programming")
    pop, log, hof = main()
    top = tools.selBest(pop, k=200)
    noDupes = []
    [noDupes.append(i) for i in top if not noDupes.count(i)]
    t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    num = len(noDupes)
    a = reporting('signal_' + t)
    for i in range(num):
        tree = gp.PrimitiveTree(noDupes[i])
        function = gp.compile(tree, pset)
        try:
            res = function(I)
            res1 = res.where(ret.isnull() == False).dropna(axis=0, how='all')
            #            idvp12rank = (-1*IDVP12).where(res1 > 0).rank(axis = 1)
            idvp20rank = (-1 * IDVP20).where(res1 > 0).rank(axis=1)
            #            totalrank = (idvp12rank.add(idvp20rank)).rank(axis = 1)
            ret_cond = ret.where(idvp20rank <= 15).dropna(axis=0, how='all')
            ret_mean = ret_cond.sum(axis=1) / 15
            a.addSectionName(str(i) + "." + "expression" + "\n" + str(tree))
            a.addParagraph("retsum:" + str(ret_mean.dropna().sum()))
            a.addParagraph("ret mean1:" + str(ret_mean.dropna().mean()) + "\n" + "IR_ret1:" + str(
                ret_mean.dropna().mean() / ret_mean.dropna().std()))
            a.addParagraph("ret mean2:" + str(ret_mean.fillna(0).mean()) + "\n" + "IR_ret2:" + str(
                ret_mean.fillna(0).mean() / ret_mean.fillna(0).std()))
            a.addDataPlot(ret_mean.fillna(0).cumsum())
            print(str(tree))
        except:
            continue
    a.turnDocx()

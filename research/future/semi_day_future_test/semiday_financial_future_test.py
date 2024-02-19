import pandas as pd
import numpy as np
import os

import model.constants.futures as ConstFut
import data.LoadLocalData.load_local_data as ExtractLocalData
import data.SQL.extract_data_from_postgre as ExtractPostgre

# tickers_list = [ConstFut.fut_simple_code_to_windcode_mapping_dict[x] for x in ConstFut.cm_cn_group_stock_index]

tickers_list = ['RB.SHF', 'HC.SHF', 'TA.CZC', 'EG.DCE', 'AL.SHF', 'CU.SHF', 'ZN.SHF', 'M.DCE', 'C.DCE', 'P.DCE',
                'Y.DCE', 'SR.CZC', 'CF.CZC']

close_price = ExtractLocalData.load_local_factor_csv(
    r'C:\Users\jason.huang\research\data_mining\GP\features\freq_4H_close.csv')
data = close_price[tickers_list].fillna(method='ffill')
data = data.dropna(how='all')
ret = data.div(data.shift(1)) - 1

total_w = pd.DataFrame()
for s, l in [(8, 16), (16, 48), (32, 96)]:

    x = data.ewm(halflife=s, adjust=True, ignore_na=True).mean() - data.ewm(halflife=l, adjust=True,
                                                                            ignore_na=True).mean()
    close_std = ret.rolling(63).std()
    y = x.div(close_std)
    z = y.div(y.rolling(252).std())
    w = z.applymap(lambda x: x / 0.89 * np.exp(-x * x / 4))

    if len(total_w) == 0:
        total_w = w.copy().mul(1 / 3)
    else:
        total_w = total_w + w.mul(1 / 3)

pnl = total_w.shift(2).mul(ret).cumsum()
pnl['total']=pnl.mean(axis=1)


pnl.to_csv("commodity_intraday.csv")
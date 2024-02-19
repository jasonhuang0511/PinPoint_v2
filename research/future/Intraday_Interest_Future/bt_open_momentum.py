# import os
# import warnings
#
# import pandas as pd
# import numpy as np
# import datetime
# import data.SQL.extract_data_from_postgre as ExtractDataPostgre
# import data.ConstantData.future_basic_information as ConstFut
# from backtest.Backtest_Object import CashBtObj, CashBt
# import pickle
# import time
#
# warnings.filterwarnings('ignore')
#
# tickers_list = ['T.CFE', 'TF.CFE', 'TS.CFE']
# bpv_value = [ConstFut.fut_code_bpv[x] for x in tickers_list]
# start_date = '2015-01-01'
# end_date = '2022-12-30'
#
# close_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
#                                                           end_date=end_date,
#                                                           key_word=['close'], freq='5min', index=1, ret_index=False,
#                                                           roll_key='gsci')
# close_price['Fut_code'] = close_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# close_price = close_price.pivot_table(index='Trade_DT', columns='Fut_code', values=close_price.columns[2])
# close_price = close_price.fillna(method='ffill')
#
# low_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
#                                                         end_date=end_date,
#                                                         key_word=['low'], freq='5min', index=1, ret_index=False,
#                                                         roll_key='gsci')
# low_price['Fut_code'] = low_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# low_price = low_price.pivot_table(index='Trade_DT', columns='Fut_code', values=low_price.columns[2])
# low_price = low_price.fillna(method='ffill')
#
# high_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
#                                                          end_date=end_date,
#                                                          key_word=['high'], freq='5min', index=1, ret_index=False,
#                                                          roll_key='gsci')
# high_price['Fut_code'] = high_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# high_price = high_price.pivot_table(index='Trade_DT', columns='Fut_code', values=high_price.columns[2])
# high_price = high_price.fillna(method='ffill')
#
# close_ret = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
#                                                         key_word=['close'], freq='5min', index=1, ret_index=True,
#                                                         roll_key='gsci')
# close_ret['Fut_code'] = close_ret['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# close_ret = close_ret.pivot_table(index='Trade_DT', columns='Fut_code', values=close_ret.columns[2])
# close_ret = close_ret.fillna(method='ffill')
#
# amount = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
#                                                      key_word=['amount'], freq='5min', index=1, ret_index=False,
#                                                      roll_key='gsci')
# amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])
# volume = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
#                                                      key_word=['volume'], freq='5min', index=1, ret_index=False,
#                                                      roll_key='gsci')
# volume['Fut_code'] = volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# volume = volume.pivot_table(index='Trade_DT', columns='Fut_code', values=volume.columns[2])
# vwap = amount.div(volume).div(bpv_value).fillna(method='ffill')
#
# common_time_index = [x for x in close_ret.index if x in close_price.index and x in vwap.index]
# close_ret = close_ret.loc[common_time_index,]
# close_price = close_price.loc[common_time_index,]
# vwap = vwap.loc[common_time_index,]
# high_price = high_price.loc[common_time_index,]
# low_price = low_price.loc[common_time_index,]
#
# ret15_min = close_ret.rolling(3).sum()
# ret120_min = close_ret.rolling(24).sum()
#
# fee_rate = 1 / 100 / 100
# aum = 10000
# mode = 'const_aum'
# trade_day_num = 252
# intraday_freq = 12 * 4
# dollar_volume_df = amount.copy()
#
# weight_df_sig1 = ret15_min.copy()
# weight_df_sig1.iloc[:, :] = 0
# for i in range(len(weight_df_sig1)):
#     if i == 0:
#         weight_df_sig1.iloc[i, :] = 0
#     elif weight_df_sig1.index[i].hour == 9 and weight_df_sig1.index[i].minute == 45:
#         weight_df_sig1.iloc[i, :] = np.sign(ret15_min.iloc[i])
#     elif weight_df_sig1.index[i].hour == 15 and weight_df_sig1.index[i].minute == 00:
#         weight_df_sig1.iloc[i, :] = 0
#     else:
#         weight_df_sig1.iloc[i, :] = weight_df_sig1.iloc[i - 1, :]
# weight_df_sig1 = weight_df_sig1.fillna(0)
#
# a1 = CashBt(weight_df=weight_df_sig1, price_df=close_price, out_of_sample_date=end_date,
#             fee_rate=fee_rate,
#             aum=aum, trade_price_df=vwap, mode=mode, trade_day_num=trade_day_num,
#             intraday_freq=intraday_freq,
#             amount_df=dollar_volume_df)
#
# weight_df_sig2 = ret15_min.copy()
# weight_df_sig2.iloc[:, :] = 0
# signal=np.sign(weight_df_sig2.iloc[0,:])
# for i in range(len(weight_df_sig2)):
#     if i == 0:
#         weight_df_sig2.iloc[i, :] = 0
#
#     if weight_df_sig2.index[i].hour == 9 and weight_df_sig2.index[i].minute == 45:
#         signal = np.sign(ret15_min.iloc[i])
#
#     if weight_df_sig2.index[i].hour == 15 and weight_df_sig2.index[i].minute == 00:
#         weight_df_sig2.iloc[i, :] = 0
#     elif weight_df_sig2.index[i].hour == 13 and weight_df_sig2.index[i].minute == 5:
#         weight_df_sig2.iloc[i, :] = signal
#     else:
#         weight_df_sig2.iloc[i, :] = weight_df_sig2.iloc[i - 1, :]
# weight_df_sig2 = weight_df_sig2.fillna(0)
# a2 = CashBt(weight_df=weight_df_sig2, price_df=close_price, out_of_sample_date=end_date,
#             fee_rate=fee_rate,
#             aum=aum, trade_price_df=vwap, mode=mode, trade_day_num=trade_day_num,
#             intraday_freq=intraday_freq,
#             amount_df=dollar_volume_df)
#
# weight_df_sig3 = ret15_min.copy()
# weight_df_sig3.iloc[:, :] = 0
# signal1 = np.sign(weight_df_sig3.iloc[0, :])
# signal2 = np.sign(weight_df_sig3.iloc[0, :])
# signal = np.sign(weight_df_sig3.iloc[0, :])
# for i in range(len(weight_df_sig3)):
#     if i == 0:
#         weight_df_sig3.iloc[i, :] = 0
#
#     if weight_df_sig3.index[i].hour == 9 and weight_df_sig3.index[i].minute == 45:
#         signal1 = np.sign(ret15_min.iloc[i])
#
#     if weight_df_sig3.index[i].hour == 9 and weight_df_sig3.index[i].minute == 30:
#         weight_df_sig3.iloc[i, :] = 0
#     elif weight_df_sig3.index[i].hour == 13 and weight_df_sig3.index[i].minute == 5:
#         signal2 = np.sign(ret120_min.iloc[i])
#         for m in range(len(signal)):
#             if signal1[m] > 0 and signal2[m] > 0:
#                 signal[m] = 1
#             elif signal1[m] < 0 and signal2[m] < 0:
#                 signal[m] = -1
#             else:
#                 signal[m] = 0
#         weight_df_sig3.iloc[i, :] = signal
#     else:
#         weight_df_sig3.iloc[i, :] = weight_df_sig3.iloc[i - 1, :]
# weight_df_sig3 = weight_df_sig3.fillna(0)
# a3 = CashBt(weight_df=weight_df_sig3, price_df=close_price, out_of_sample_date=end_date,
#             fee_rate=fee_rate,
#             aum=aum, trade_price_df=vwap, mode=mode, trade_day_num=trade_day_num,
#             intraday_freq=intraday_freq,
#             amount_df=dollar_volume_df)
#
# weight_df_sig4 = ret15_min.copy()
# weight_df_sig4.iloc[:, :] = 0
# signal1 = np.sign(weight_df_sig4.iloc[0, :])
# signal2 = np.sign(weight_df_sig4.iloc[0, :])
# signal = np.sign(weight_df_sig4.iloc[0, :])
# for i in range(len(weight_df_sig4)):
#     if i == 0:
#         weight_df_sig4.iloc[i, :] = 0
#
#     if weight_df_sig4.index[i].hour == 9 and weight_df_sig4.index[i].minute == 45:
#         signal1 = np.sign(weight_df_sig4.iloc[0, :])
#         for m in range(len(signal1)):
#             if low_price.iloc[i, m] >= low_price.iloc[i - 1, m] >= low_price.iloc[i - 2, m]:
#                 signal1[m] = 1
#             elif high_price.iloc[i - 2, m] >= high_price.iloc[i - 1, m] >= high_price.iloc[i, m]:
#                 signal1[m] = -1
#             else:
#                 signal1[m] = 0
#         # signal1 = np.sign(ret15_min.iloc[i])
#
#     if weight_df_sig4.index[i].hour == 9 and weight_df_sig4.index[i].minute == 30:
#         weight_df_sig4.iloc[i, :] = 0
#     elif weight_df_sig4.index[i].hour == 13 and weight_df_sig4.index[i].minute == 5:
#         signal2 = np.sign(ret120_min.iloc[i])
#         for m in range(len(signal)):
#             if signal1[m] > 0 and signal2[m] > 0:
#                 signal[m] = 1
#             elif signal1[m] < 0 and signal2[m] < 0:
#                 signal[m] = -1
#             else:
#                 signal[m] = 0
#         weight_df_sig4.iloc[i, :] = signal
#     else:
#         weight_df_sig4.iloc[i, :] = weight_df_sig4.iloc[i - 1, :]
#
# weight_df_sig4 = weight_df_sig4.fillna(0)
# a4 = CashBt(weight_df=weight_df_sig4, price_df=close_price, out_of_sample_date=end_date,
#             fee_rate=fee_rate,
#             aum=aum, trade_price_df=vwap, mode=mode, trade_day_num=trade_day_num,
#             intraday_freq=intraday_freq,
#             amount_df=dollar_volume_df)
#
# file_path='C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\'
# with open(f"{file_path}signal_1_{start_date}_{end_date}.pkl",'wb') as file:
#     pickle.dump(a1, file, pickle.HIGHEST_PROTOCOL)
# with open(f"{file_path}signal_2_{start_date}_{end_date}.pkl",'wb') as file:
#     pickle.dump(a2, file, pickle.HIGHEST_PROTOCOL)
# with open(f"{file_path}signal_3_{start_date}_{end_date}.pkl",'wb') as file:
#     pickle.dump(a3, file, pickle.HIGHEST_PROTOCOL)
# with open(f"{file_path}signal_4_{start_date}_{end_date}.pkl",'wb') as file:
#     pickle.dump(a4, file, pickle.HIGHEST_PROTOCOL)


import os
import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut
from backtest.Backtest_Object import CashBtObj, CashBt
import pickle
import time

warnings.filterwarnings('ignore')

date_split = datetime.date(2020, 7, 20)
record_rule_change_date=datetime.date(2021,9,14)


def process_pnl_annually(pnl_df_data):
    pnl_df = pnl_df_data.copy()
    tickers_list = pnl_df.columns
    pnl_df['Year'] = [x.year for x in pnl_df.index]
    annual_ret = pnl_df.groupby(['Year'])[tickers_list].mean().mul(252)
    annual_ret.loc['All', :] = pnl_df[tickers_list].mean().mul(252)
    annual_std = pnl_df.groupby(['Year'])[tickers_list].std().mul(np.sqrt(252))
    annual_std.loc['All', :] = pnl_df[tickers_list].std().mul(np.sqrt(252))
    annual_ir = annual_ret.div(annual_std)
    annual_mdd = annual_ret.copy()
    annual_win_ratio = annual_ret.copy()
    for i in range(len(annual_mdd)):
        for j in range(len(annual_mdd.columns)):
            year = annual_mdd.index[i]
            tickers = annual_mdd.columns[j]
            if year == 'All':
                data = pnl_df[tickers]
            else:
                data = pnl_df[pnl_df['Year'] == year]
                data = data[tickers]
            data1 = data.copy()
            data1 = data1.replace(0, np.nan)
            data1 = data1.dropna()
            try:
                annual_win_ratio.iloc[i, j] = len(data1[data1 > 0]) / len(data1)
            except:
                annual_win_ratio.iloc[i, j] = np.nan
            data = data.fillna(0)
            try:
                annual_mdd.iloc[i, j] = np.abs((data.cumsum() - data.cumsum().cummax()).min())
            except:
                annual_mdd.iloc[i, j] = np.nan
    annual_calmar = annual_ret.div(annual_mdd)
    result_dict = {}
    for tickers_name in tickers_list:
        df = pd.concat(
            [annual_ret[tickers_name], annual_std[tickers_name], annual_mdd[tickers_name], annual_ir[tickers_name],
             annual_calmar[tickers_name], annual_win_ratio[tickers_name]], axis=1)
        df.columns = ['Ret', 'Std', 'MDD', 'IR', 'Calmar', 'Win_Ratio']
        result_dict[tickers_name] = df
    return result_dict


tickers_list = ['T.CFE', 'TF.CFE', 'TS.CFE']
bpv_value = [ConstFut.fut_code_bpv[x] for x in tickers_list]
start_date = '2015-01-01'
end_date = '2022-12-01'
roll_method = 'trade_vol'

# minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
#                                                            end_date=end_date,
#                                                            key_word=['open'], freq='1min', index=1, ret_index=False,
#                                                            roll_method=roll_method)
#
# minute_price['Fut_code'] = minute_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
# minute_price['Date'] = [x.date() for x in minute_price['Trade_DT']]
# minute_price['Time'] = [x.time() for x in minute_price['Trade_DT']]
# df_all = pd.DataFrame()
# for tickers_name in np.unique(np.array(minute_price['Fut_code'])):
#     df_tickers = minute_price[minute_price['Fut_code'] == tickers_name]
#     df_tickers = df_tickers.sort_values(['Trade_DT'])
#     df_tickers = df_tickers.fillna(method='ffill')
#     df_all = pd.concat([df_all, df_tickers])
# open_minute_price = df_all.copy()

minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['amount'], freq='1min', index=1, ret_index=False,
                                                           roll_method=roll_method)

minute_price['Fut_code'] = minute_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
minute_price['Date'] = [x.date() for x in minute_price['Trade_DT']]
minute_price['Time'] = [x.time() for x in minute_price['Trade_DT']]
df_all = pd.DataFrame()
for tickers_name in np.unique(np.array(minute_price['Fut_code'])):
    df_tickers = minute_price[minute_price['Fut_code'] == tickers_name]
    df_tickers = df_tickers.sort_values(['Trade_DT'])
    df_tickers = df_tickers.fillna(method='ffill')
    df_all = pd.concat([df_all, df_tickers])
amount_minute = df_all.copy()

minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['volume'], freq='1min', index=1, ret_index=False,
                                                           roll_method=roll_method)

minute_price['Fut_code'] = minute_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
minute_price['Date'] = [x.date() for x in minute_price['Trade_DT']]
minute_price['Time'] = [x.time() for x in minute_price['Trade_DT']]
df_all = pd.DataFrame()
for tickers_name in np.unique(np.array(minute_price['Fut_code'])):
    df_tickers = minute_price[minute_price['Fut_code'] == tickers_name]
    df_tickers = df_tickers.sort_values(['Trade_DT'])
    df_tickers = df_tickers.fillna(method='ffill')
    df_all = pd.concat([df_all, df_tickers])
volume_minute = df_all.copy()

vwap_minute = pd.merge(left=amount_minute, right=volume_minute, on=['Code', 'Trade_DT', 'Fut_code', 'Date', 'Time'],
                       how='outer')

datetime_list = [datetime.time(15, 11), datetime.time(15, 12), datetime.time(15, 13), datetime.time(15, 14),
                 datetime.time(15, 15)]


def get_ts_vwap(vwap_minute, datetime_list):
    volume_all = pd.DataFrame()
    amount_all = pd.DataFrame()
    for datetime_minute in datetime_list:
        volume_one_time = vwap_minute[vwap_minute['Time'] == datetime_minute][['Fut_code', 'Date', 'volume']]
        amount_one_time = vwap_minute[vwap_minute['Time'] == datetime_minute][['Fut_code', 'Date', 'amount']]
        if len(volume_all) == 0:
            volume_all = volume_one_time
        else:
            volume_all = pd.merge(left=volume_all, right=volume_one_time, on=['Fut_code', 'Date'])
        if len(amount_all) == 0:
            amount_all = amount_one_time
        else:
            amount_all = pd.merge(left=amount_all, right=amount_one_time, on=['Fut_code', 'Date'])
    volume_all = volume_all.fillna(0)
    volume_all['volume_sum'] = volume_all.iloc[:, 2:].sum(axis=1)
    volume_all = volume_all[['Fut_code', 'Date', 'volume_sum']]
    amount_all = amount_all.fillna(0)
    amount_all['amount_sum'] = amount_all.iloc[:, 2:].sum(axis=1)
    amount_all = amount_all[['Fut_code', 'Date', 'amount_sum']]
    vwap_all = pd.merge(left=amount_all, right=volume_all, on=['Fut_code', 'Date'], how='outer')
    vwap_all['bpv'] = [ConstFut.fut_code_bpv[x] for x in vwap_all['Fut_code']]
    vwap_all['vwap'] = vwap_all['amount_sum'].div(vwap_all['volume_sum']).div(vwap_all['bpv'])
    vwap_all = vwap_all.pivot_table(index='Date', columns='Fut_code', values='vwap')
    return vwap_all


vwap_0931_0935 = get_ts_vwap(vwap_minute,
                             [datetime.time(9, 30), datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                              datetime.time(9, 34)])
vwap_0945_0949 = get_ts_vwap(vwap_minute,
                             [datetime.time(9, 45), datetime.time(9, 46), datetime.time(9, 47), datetime.time(9, 48),
                              datetime.time(9, 49)])
vwap_morning_revise = pd.concat([vwap_0931_0935[vwap_0931_0935.index < date_split],
                                 vwap_0945_0949[vwap_0945_0949.index >= date_split]])

vwap_1330_1334 = get_ts_vwap(vwap_minute,
                             [datetime.time(13, 30), datetime.time(13, 31), datetime.time(13, 32),
                              datetime.time(13, 33),
                              datetime.time(13, 34)])
vwap_1511_1515 = get_ts_vwap(vwap_minute,
                             [datetime.time(15, 11), datetime.time(15, 12), datetime.time(15, 13),
                              datetime.time(15, 14),
                              datetime.time(15, 15)])

minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['close'], freq='1min', index=1, ret_index=False,
                                                           roll_method=roll_method)

minute_price['Fut_code'] = minute_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
minute_price['Date'] = [x.date() for x in minute_price['Trade_DT']]
minute_price['Time'] = [x.time() for x in minute_price['Trade_DT']]
df_all = pd.DataFrame()
for tickers_name in np.unique(np.array(minute_price['Fut_code'])):
    df_tickers = minute_price[minute_price['Fut_code'] == tickers_name]
    df_tickers = df_tickers.sort_values(['Trade_DT'])
    df_tickers = df_tickers.fillna(method='ffill')
    df_all = pd.concat([df_all, df_tickers])
close_minute_price = df_all.copy()

# 前15分钟的close
morning_price_part1 = close_minute_price[close_minute_price['Time'] == datetime.time(9, 29)]
morning_price_part1 = morning_price_part1.pivot_table(index='Date', columns='Fut_code',
                                                      values=morning_price_part1.columns[2])
morning_price_part2 = close_minute_price[close_minute_price['Time'] == datetime.time(9, 44)]
morning_price_part2 = morning_price_part2.pivot_table(index='Date', columns='Fut_code',
                                                      values=morning_price_part2.columns[2])
morning_price = pd.concat([morning_price_part1[morning_price_part1.index < date_split],
                           morning_price_part2[morning_price_part2.index >= date_split]])

# 下午13：30时刻的price
afternoon_price = close_minute_price[close_minute_price['Time'] == datetime.time(13, 29)]
afternoon_price = afternoon_price.pivot_table(index='Date', columns='Fut_code', values=afternoon_price.columns[2])

# 中午11：30时刻的price
noon_price = close_minute_price[close_minute_price['Time'] == datetime.time(11, 29)]
noon_price = noon_price.pivot_table(index='Date', columns='Fut_code', values=noon_price.columns[2])

# open_price = minute_price[minute_price['Time'] == datetime.time(9, 30)]
# open_price = open_price.pivot_table(index='Date', columns='Fut_code', values=open_price.columns[2])

# 开盘价
open_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                         end_date=end_date,
                                                         key_word=['open'], freq='Daily', index=1, ret_index=False,
                                                         roll_method=roll_method)
open_price['Fut_code'] = open_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open_price = open_price.pivot_table(index='Trade_DT', columns='Fut_code', values=open_price.columns[2])

# vwap 5min
amount = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word=['amount'], freq='5min', index=1, ret_index=False,
                                                     roll_method=roll_method)
amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])
volume = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word=['volume'], freq='5min', index=1, ret_index=False,
                                                     roll_method=roll_method)
volume['Fut_code'] = volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume = volume.pivot_table(index='Trade_DT', columns='Fut_code', values=volume.columns[2])
vwap = amount.div(volume).div(bpv_value).fillna(method='ffill')

vwap['Date'] = [x.date() for x in vwap.index]
vwap['Time'] = [x.time() for x in vwap.index]

# 13：00-13：05 vwap
vwap_afternoon_1305 = vwap[vwap['Time'] == datetime.time(13, 5)]
vwap_afternoon_1305.index = vwap_afternoon_1305['Date']
vwap_afternoon_1305 = vwap_afternoon_1305.iloc[:, :-2]

# 13：05-13：10 vwap
vwap_afternoon_1310 = vwap[vwap['Time'] == datetime.time(13, 10)]
vwap_afternoon_1310.index = vwap_afternoon_1310['Date']
vwap_afternoon_1310 = vwap_afternoon_1310.iloc[:, :-2]

# 13：30-13：35 vwap
vwap_afternoon_1335 = vwap[vwap['Time'] == datetime.time(13, 35)]
vwap_afternoon_1335.index = vwap_afternoon_1335['Date']
vwap_afternoon_1335 = vwap_afternoon_1335.iloc[:, :-2]

# 15：10-15：15 vwap
vwap_close = vwap[vwap['Time'] == datetime.time(15, 15)]
vwap_close.index = vwap_close['Date']
vwap_close = vwap_close.iloc[:, :-2]

# 开盘15分钟后，后5分钟vwap
vwap_morning_part1 = vwap[vwap['Time'] == datetime.time(9, 35)]
vwap_morning_part1.index = vwap_morning_part1['Date']
vwap_morning_part1 = vwap_morning_part1.iloc[:, :-2]
vwap_morning_part2 = vwap[vwap['Time'] == datetime.time(9, 50)]
vwap_morning_part2.index = vwap_morning_part2['Date']
vwap_morning_part2 = vwap_morning_part2.iloc[:, :-2]
vwap_morning = pd.concat([vwap_morning_part1[vwap_morning_part1.index < date_split],
                          vwap_morning_part2[vwap_morning_part2.index >= date_split]])

weight_df_1 = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_1 = vwap_close.div(vwap_morning).sub(1)

nv_df_1 = weight_df_1.mul(ret_df_1).cumsum()
pnl_df_1 = weight_df_1.mul(ret_df_1)
a = process_pnl_annually(pnl_df_1)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\indicator_df_1{roll_method}.csv")

nv_df_1.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\nv_df_1_{roll_method}.csv")

# signal 2
weight_df_2 = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_2 = vwap_close.div(vwap_afternoon_1305).sub(1)

nv_df_2 = weight_df_2.mul(ret_df_2).cumsum()
pnl_df_2 = weight_df_2.mul(ret_df_2)

a = process_pnl_annually(pnl_df_2)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\indicator_df_2{roll_method}.csv")

nv_df_2.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\nv_df_2_{roll_method}.csv")

# signal 2 open out
weight_df_2_overnight = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_2_overnight = open_price.shift(-1).div(vwap_afternoon_1305).sub(1)

nv_df_2_overnight = weight_df_2_overnight.mul(ret_df_2_overnight).cumsum()
pnl_df_2_overnight = weight_df_2_overnight.mul(ret_df_2_overnight)
a = process_pnl_annually(pnl_df_2_overnight)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\indicator_df_2_overnight_{roll_method}.csv")

nv_df_2_overnight.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\nv_df_2_overnight_{roll_method}.csv")

# signal 3
weight_df_3_1 = morning_price.div(open_price).sub(1).applymap(np.sign)
weight_df_3_2 = afternoon_price.div(open_price).sub(1).applymap(np.sign)
weight_df_3 = weight_df_3_1.copy()
weight_df_3.iloc[:, :] = 0
for i in range(len(weight_df_3)):
    for j in range(len(weight_df_3.columns)):
        if weight_df_3_1.iloc[i, j] > 0 and weight_df_3_2.iloc[i, j] >= 0:
            weight_df_3.iloc[i, j] = 1
        elif weight_df_3_1.iloc[i, j] < 0 and weight_df_3_2.iloc[i, j] <= 0:
            weight_df_3.iloc[i, j] = -1
        else:
            weight_df_3.iloc[i, j] = 0

ret_df_3 = open_price.shift(-1).div(vwap_afternoon_1310).sub(1)

nv_df_3 = weight_df_3.mul(ret_df_3).cumsum()
pnl_df_3 = weight_df_3.mul(ret_df_3)

a = process_pnl_annually(pnl_df_3)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\indicator_df_2_overnight_{roll_method}.csv")

nv_df_3.to_csv(
    f"C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\nv_df_3_{roll_method}.csv")

############################
# new vwap


weight_df_1 = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_1 = vwap_1511_1515.div(vwap_morning_revise).sub(1)

nv_df_1 = weight_df_1.mul(ret_df_1).cumsum()
pnl_df_1 = weight_df_1.mul(ret_df_1)
a = process_pnl_annually(pnl_df_1)

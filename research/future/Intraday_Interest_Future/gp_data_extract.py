import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

import os

warnings.filterwarnings('ignore')

change_trading_time_start_date = datetime.date(2020, 7, 20)
database_change_record_rule_start_date = datetime.date(2021, 9, 15)


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


def get_ts_vwap(vwap_minute_df, datetime_time_list):
    vwap_minute = vwap_minute_df.copy()
    datetime_list = datetime_time_list
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


def get_ts_high(high_minute_df, datetime_time_list):
    high_all = pd.DataFrame()
    for datetime_minute in datetime_time_list:
        high_one_time = high_minute_df[high_minute_df['Time'] == datetime_minute][['Fut_code', 'Date', 'high']]

        if len(high_all) == 0:
            high_all = high_one_time
        else:
            high_all = pd.merge(left=high_all, right=high_one_time, on=['Fut_code', 'Date'])

    high_all['highest'] = high_all.iloc[:, 2:].max(axis=1)
    high_all = high_all.pivot_table(index='Date', columns='Fut_code', values='highest')
    return high_all


def get_ts_low(low_minute_df, datetime_time_list):
    low_all = pd.DataFrame()
    for datetime_minute in datetime_time_list:
        low_one_time = low_minute_df[low_minute_df['Time'] == datetime_minute][['Fut_code', 'Date', 'low']]

        if len(low_all) == 0:
            low_all = low_one_time
        else:
            low_all = pd.merge(left=low_all, right=low_one_time, on=['Fut_code', 'Date'])

    low_all['lowest'] = low_all.iloc[:, 2:].min(axis=1)
    low_all = low_all.pivot_table(index='Date', columns='Fut_code', values='lowest')
    return low_all


file_save_path = "C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\GP_data\\"
tickers_list = ['T.CFE', 'TF.CFE', 'TS.CFE']
bpv_value = [ConstFut.fut_code_bpv[x] for x in tickers_list]
start_date = '2020-07-01'
end_date = '2022-12-01'
roll_method = 'oi'

# 开盘价
daily_feature_list = ['open', 'high', 'low', 'close','amount','volume']
for daily_feature in daily_feature_list:
    open_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                             end_date=end_date,
                                                             key_word=[daily_feature], freq='Daily', index=1,
                                                             ret_index=False,
                                                             roll_method=roll_method)
    open_price['Fut_code'] = open_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
    open_price = open_price.pivot_table(index='Trade_DT', columns='Fut_code', values=open_price.columns[2])
    if daily_feature == 'open':
        open_price.to_csv(f"{file_save_path}freq_1D_todayopen.csv")
    open_price = open_price.shift(1).iloc[1:, :]
    open_price.index.name = 'Trade_DT'
    open_price.to_csv(f"{file_save_path}freq_1D_pre{daily_feature}.csv")
    print(f"{daily_feature} is ok")

intraday_feature_list = ['open', 'high', 'low', 'close', 'volume', 'twap', 'position_range', 't_vwap']
datetime_list = ['0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', '1315']
for intraday_feature in intraday_feature_list:
    minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                               end_date=end_date,
                                                               key_word=[intraday_feature], freq='15min', index=1,
                                                               ret_index=False,
                                                               roll_method=roll_method)
    minute_price['Fut_code'] = minute_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
    minute_price['Date'] = [x.date() for x in minute_price['Trade_DT']]
    minute_price['Time'] = [x.time() for x in minute_price['Trade_DT']]
    for time_str in datetime_list:
        t = datetime.time(int(time_str[:2]), int(time_str[2:]))
        df_minute = minute_price[minute_price['Time'] == t]
        df_minute = df_minute.pivot_table(values=df_minute.columns[2], index='Date', columns='Fut_code')
        df_minute = df_minute.iloc[1:, :]
        df_minute.index.name = 'Trade_DT'
        df_minute.to_csv(f"{file_save_path}freq_1D_{intraday_feature}{time_str}.csv")
        print(f"{intraday_feature}   {time_str}  is ok")

# vwap minute data
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

# vwap close last 5th vwap
vwap_close_price_part_1 = get_ts_vwap(vwap_minute,
                                      [datetime.time(15, 11), datetime.time(15, 12), datetime.time(15, 13),
                                       datetime.time(15, 14), datetime.time(15, 15)])
vwap_close_price_part_1 = vwap_close_price_part_1[
    vwap_close_price_part_1.index < database_change_record_rule_start_date]
vwap_close_price_part_2 = get_ts_vwap(vwap_minute,
                                      [datetime.time(15, 10), datetime.time(15, 11), datetime.time(15, 12),
                                       datetime.time(15, 13), datetime.time(15, 14)])
vwap_close_price_part_2 = vwap_close_price_part_2[
    vwap_close_price_part_2.index >= database_change_record_rule_start_date]
vwap_close = pd.concat([vwap_close_price_part_1, vwap_close_price_part_2])

# vwap afternoon 30th-35th vwap
vwap_afternoon_part_1 = get_ts_vwap(vwap_minute,
                                    [datetime.time(13, 31), datetime.time(13, 32), datetime.time(13, 33),
                                     datetime.time(13, 34), datetime.time(13, 35)])
vwap_afternoon_part_1 = vwap_afternoon_part_1[
    vwap_afternoon_part_1.index < database_change_record_rule_start_date]
vwap_afternoon_part_2 = get_ts_vwap(vwap_minute,
                                    [datetime.time(13, 30), datetime.time(13, 11), datetime.time(13, 32),
                                     datetime.time(13, 33), datetime.time(13, 34)])
vwap_afternoon_part_2 = vwap_afternoon_part_2[
    vwap_afternoon_part_2.index >= database_change_record_rule_start_date]
vwap_afternoon = pd.concat([vwap_afternoon_part_1, vwap_afternoon_part_2])

ret_df = vwap_close.div(vwap_afternoon) - 1
ret_df.index.name = 'Trade_DT'
ret_df.to_csv(f"{file_save_path}freq_1D_close_ret.csv")

for file in os.listdir(file_save_path):
    data = pd.read_csv(file_save_path + file, index_col=0)
    data.index.name = 'Trade_DT'
    print(data.shape)
    data.to_csv()

feature_name_list = [x[8:-4] for x in os.listdir(file_save_path)]

feature_name_list_str = ['close0945', 'close1000', 'close1015', 'close1030',
                         'close1045', 'close1100', 'close1115', 'close1130',
                         'close1315', 'close_ret', 'high0945', 'high1000',
                         'high1015', 'high1030', 'high1045', 'high1100',
                         'high1115', 'high1130', 'high1315', 'low0945',
                         'low1000', 'low1015', 'low1030', 'low1045',
                         'low1100', 'low1115', 'low1130', 'low1315',
                         'open0945', 'open1000', 'open1015', 'open1030',
                         'open1045', 'open1100', 'open1115', 'open1130',
                         'open1315', 'position_range0945', 'position_range1000', 'position_range1015',
                         'position_range1030', 'position_range1045', 'position_range1100', 'position_range1115',
                         'position_range1130', 'position_range1315', 'preclose', 'prehigh',
                         'prelow', 'preopen', 'todayopen', 'twap0945',
                         'twap1000', 'twap1015', 'twap1030', 'twap1045',
                         'twap1100', 'twap1115', 'twap1130', 'twap1315',
                         't_vwap0945', 't_vwap1000', 't_vwap1015', 't_vwap1030',
                         't_vwap1045', 't_vwap1100', 't_vwap1115', 't_vwap1130',
                         't_vwap1315', 'volume0945', 'volume1000', 'volume1015',
                         'volume1030', 'volume1045', 'volume1100', 'volume1115', 'volume1130', 'volume1315']




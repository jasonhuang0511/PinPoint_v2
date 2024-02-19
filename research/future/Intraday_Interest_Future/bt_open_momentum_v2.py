import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut
from scipy import optimize

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
    annual_turnover = annual_ret.copy()
    for i in range(len(annual_mdd)):
        for j in range(len(annual_mdd.columns)):
            year = annual_mdd.index[i]
            tickers = annual_mdd.columns[j]
            if year == 'All':
                data = pnl_df[tickers]
            else:
                data = pnl_df[pnl_df['Year'] == year]
                data = data[tickers]

            # win ratio
            data1 = data.copy()
            data1 = data1.replace(0, np.nan)
            data1 = data1.dropna()
            try:
                annual_win_ratio.iloc[i, j] = len(data1[data1 > 0]) / len(data1)
            except:
                annual_win_ratio.iloc[i, j] = np.nan
            # turnover
            data2 = data.copy()
            data2 = data2.dropna()
            try:
                annual_turnover.iloc[i, j] = (1 - len(data2[data2 == 0]) / len(data2)) * 2
            except:
                annual_turnover.iloc[i, j] = np.nan

            # mdd
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
             annual_calmar[tickers_name], annual_win_ratio[tickers_name], annual_turnover[tickers_name]], axis=1)
        df.columns = ['Ret', 'Std', 'MDD', 'IR', 'Calmar', 'Win_Ratio', 'Turnover']
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


file_save_path = "C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\tick\\"
tickers_list = ['T.CFE', 'TF.CFE', 'TS.CFE']
bpv_value = [ConstFut.fut_code_bpv[x] for x in tickers_list]
start_date = '2015-01-01'
end_date = '2022-12-01'
roll_method = 'oi'

# high price minute data
minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['high'], freq='1min', index=1, ret_index=False,
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
high_minute_price = df_all.copy()

high_minute_part_1 = get_ts_high(high_minute_price, [datetime.time(9, 16), datetime.time(9, 17), datetime.time(9, 18),
                                                     datetime.time(9, 19), datetime.time(9, 20)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_high(high_minute_price, [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                                     datetime.time(9, 34), datetime.time(9, 35)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_high(high_minute_price, [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                                     datetime.time(9, 34), datetime.time(9, 35)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
high_bar_1 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_high(high_minute_price, [datetime.time(9, 21), datetime.time(9, 22), datetime.time(9, 23),
                                                     datetime.time(9, 24), datetime.time(9, 25)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_high(high_minute_price, [datetime.time(9, 36), datetime.time(9, 37), datetime.time(9, 38),
                                                     datetime.time(9, 39), datetime.time(9, 40)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_high(high_minute_price, [datetime.time(9, 36), datetime.time(9, 37), datetime.time(9, 38),
                                                     datetime.time(9, 39), datetime.time(9, 40)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
high_bar_2 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_high(high_minute_price, [datetime.time(9, 26), datetime.time(9, 27), datetime.time(9, 28),
                                                     datetime.time(9, 29), datetime.time(9, 30)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_high(high_minute_price, [datetime.time(9, 41), datetime.time(9, 42), datetime.time(9, 43),
                                                     datetime.time(9, 44), datetime.time(9, 45)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_high(high_minute_price, [datetime.time(9, 41), datetime.time(9, 42), datetime.time(9, 43),
                                                     datetime.time(9, 44), datetime.time(9, 45)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
high_bar_3 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

# low price minute data
minute_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['low'], freq='1min', index=1, ret_index=False,
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
high_minute_price = df_all.copy()

high_minute_part_1 = get_ts_low(high_minute_price, [datetime.time(9, 16), datetime.time(9, 17), datetime.time(9, 18),
                                                    datetime.time(9, 19), datetime.time(9, 20)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_low(high_minute_price, [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                                    datetime.time(9, 34), datetime.time(9, 35)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_low(high_minute_price,  [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                                    datetime.time(9, 34), datetime.time(9, 35)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
low_bar_1 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_low(high_minute_price, [datetime.time(9, 21), datetime.time(9, 22), datetime.time(9, 23),
                                                    datetime.time(9, 24), datetime.time(9, 25)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_low(high_minute_price, [datetime.time(9, 36), datetime.time(9, 37), datetime.time(9, 38),
                                                    datetime.time(9, 39), datetime.time(9, 40)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_low(high_minute_price, [datetime.time(9, 36), datetime.time(9, 37), datetime.time(9, 38),
                                                    datetime.time(9, 39), datetime.time(9, 40)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
low_bar_2 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_low(high_minute_price, [datetime.time(9, 26), datetime.time(9, 27), datetime.time(9, 28),
                                                    datetime.time(9, 29), datetime.time(9, 30)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_low(high_minute_price, [datetime.time(9, 41), datetime.time(9, 42), datetime.time(9, 43),
                                                    datetime.time(9, 44), datetime.time(9, 45)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_low(high_minute_price,[datetime.time(9, 41), datetime.time(9, 42), datetime.time(9, 43),
                                                    datetime.time(9, 44), datetime.time(9, 45)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
low_bar_3 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

# 开盘价
open_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                         end_date=end_date,
                                                         key_word=['open'], freq='Daily', index=1, ret_index=False,
                                                         roll_method=roll_method)
open_price['Fut_code'] = open_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open_price = open_price.pivot_table(index='Trade_DT', columns='Fut_code', values=open_price.columns[2])
# close
close_price = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                          end_date=end_date,
                                                          key_word=['close'], freq='Daily', index=1, ret_index=False,
                                                          roll_method=roll_method)
close_price['Fut_code'] = close_price['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close_price = close_price.pivot_table(index='Trade_DT', columns='Fut_code', values=close_price.columns[2])

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
morning_price_part1 = close_minute_price[close_minute_price['Time'] == datetime.time(9, 30)]
morning_price_part1 = morning_price_part1.pivot_table(index='Date', columns='Fut_code',
                                                      values=morning_price_part1.columns[2])
morning_price_part1 = morning_price_part1[morning_price_part1.index < change_trading_time_start_date]

morning_price_part2 = close_minute_price[close_minute_price['Time'] == datetime.time(9, 45)]
morning_price_part2 = morning_price_part2.pivot_table(index='Date', columns='Fut_code',
                                                      values=morning_price_part2.columns[2])
morning_price_part2 = morning_price_part2[morning_price_part2.index >= change_trading_time_start_date]
morning_price_part2 = morning_price_part2[morning_price_part2.index < database_change_record_rule_start_date]

morning_price_part3 = close_minute_price[close_minute_price['Time'] == datetime.time(9, 45)]
morning_price_part3 = morning_price_part3.pivot_table(index='Date', columns='Fut_code',
                                                      values=morning_price_part3.columns[2])
morning_price_part3 = morning_price_part3[morning_price_part3.index >= database_change_record_rule_start_date]

morning_price = pd.concat([morning_price_part1, morning_price_part2, morning_price_part3])

# 下午13：30时刻的price
afternoon_price_part_1 = close_minute_price[close_minute_price['Time'] == datetime.time(13, 30)]
afternoon_price_part_1 = afternoon_price_part_1.pivot_table(index='Date', columns='Fut_code',
                                                            values=afternoon_price_part_1.columns[2])
afternoon_price_part_1 = afternoon_price_part_1[afternoon_price_part_1.index < database_change_record_rule_start_date]

afternoon_price_part_2 = close_minute_price[close_minute_price['Time'] == datetime.time(13, 29)]
afternoon_price_part_2 = afternoon_price_part_2.pivot_table(index='Date', columns='Fut_code',
                                                            values=afternoon_price_part_2.columns[2])
afternoon_price_part_2 = afternoon_price_part_2[afternoon_price_part_2.index >= database_change_record_rule_start_date]

afternoon_price = pd.concat([afternoon_price_part_1, afternoon_price_part_2])

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

# morning vwap 15th-20th
vwap_morning_part_1 = get_ts_vwap(vwap_minute,
                                  [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                   datetime.time(9, 34), datetime.time(9, 35)])
vwap_morning_part_1 = vwap_morning_part_1[vwap_morning_part_1.index < change_trading_time_start_date]

vwap_morning_part_2 = get_ts_vwap(vwap_minute,
                                  [datetime.time(9, 46), datetime.time(9, 47), datetime.time(9, 48),
                                   datetime.time(9, 49), datetime.time(9, 50)])
vwap_morning_part_2 = vwap_morning_part_2[vwap_morning_part_2.index >= change_trading_time_start_date]
vwap_morning_part_2 = vwap_morning_part_2[vwap_morning_part_2.index < database_change_record_rule_start_date]

vwap_morning_part_3 = get_ts_vwap(vwap_minute,
                                  [datetime.time(9, 46), datetime.time(9, 47), datetime.time(9, 48),
                                   datetime.time(9, 49), datetime.time(9, 50)])
vwap_morning_part_3 = vwap_morning_part_3[vwap_morning_part_3.index >= database_change_record_rule_start_date]

vwap_morning = pd.concat([vwap_morning_part_1, vwap_morning_part_2, vwap_morning_part_3])

# vwap close last 5th vwap
vwap_close_price_part_1 = get_ts_vwap(vwap_minute,
                                      [datetime.time(15, 11), datetime.time(15, 12), datetime.time(15, 13),
                                       datetime.time(15, 14), datetime.time(15, 15)])
vwap_close_price_part_1 = vwap_close_price_part_1[
    vwap_close_price_part_1.index < database_change_record_rule_start_date]
vwap_close_price_part_2 = get_ts_vwap(vwap_minute,
                                      [datetime.time(15, 11), datetime.time(15, 12), datetime.time(15, 13),
                                       datetime.time(15, 14), datetime.time(15, 15)])
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
                                    [datetime.time(13, 31), datetime.time(13, 32), datetime.time(13, 33),
                                     datetime.time(13, 34), datetime.time(13, 35)])
vwap_afternoon_part_2 = vwap_afternoon_part_2[
    vwap_afternoon_part_2.index >= database_change_record_rule_start_date]
vwap_afternoon = pd.concat([vwap_afternoon_part_1, vwap_afternoon_part_2])

####################################################################
# signal 1
weight_df_1 = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_1 = vwap_close.div(vwap_morning).sub(1)

nv_df_1 = weight_df_1.mul(ret_df_1).fillna(0).cumsum()
pnl_df_1 = weight_df_1.mul(ret_df_1)
a = process_pnl_annually(pnl_df_1)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_1{roll_method}.csv")

nv_df_1.to_csv(f"{file_save_path}nv_df_1_{roll_method}.csv")

# signal 2
weight_df_2 = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_2 = vwap_close.div(vwap_afternoon).sub(1)

nv_df_2 = weight_df_2.mul(ret_df_2).fillna(0).cumsum()
pnl_df_2 = weight_df_2.mul(ret_df_2)

a = process_pnl_annually(pnl_df_2)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_2{roll_method}.csv")

nv_df_2.to_csv(f"{file_save_path}nv_df_2_{roll_method}.csv")

# signal 2 open out
weight_df_2_overnight = morning_price.div(open_price).sub(1).applymap(np.sign)
ret_df_2_overnight = open_price.shift(-1).div(vwap_afternoon).sub(1)

nv_df_2_overnight = weight_df_2_overnight.mul(ret_df_2_overnight).fillna(0).cumsum()
pnl_df_2_overnight = weight_df_2_overnight.mul(ret_df_2_overnight)
a = process_pnl_annually(pnl_df_2_overnight)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_2_overnight_{roll_method}.csv")

nv_df_2_overnight.to_csv(f"{file_save_path}nv_df_2_overnight_{roll_method}.csv")

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

ret_df_3 = open_price.shift(-1).div(vwap_afternoon).sub(1)

nv_df_3 = weight_df_3.mul(ret_df_3).fillna(0).cumsum()
pnl_df_3 = weight_df_3.mul(ret_df_3)

a = process_pnl_annually(pnl_df_3)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_{roll_method}.csv")

nv_df_3.to_csv(f"{file_save_path}nv_df_3_{roll_method}.csv")

ret_df_3_after_fee = open_price.shift(-1).div(vwap_afternoon).sub(1).sub(3 / 100 / 100 + 3 / 100 / 10000)

nv_df_3_after_fee = weight_df_3.mul(ret_df_3_after_fee).cumsum()
pnl_df_3_after_fee = weight_df_3.mul(ret_df_3_after_fee)

a = process_pnl_annually(pnl_df_3_after_fee)
############################
# signal 3 only short or only long

weight_df_3_1 = morning_price.div(open_price).sub(1).applymap(np.sign)
weight_df_3_2 = afternoon_price.div(open_price).sub(1).applymap(np.sign)
weight_df_3_short = weight_df_3_1.copy()
weight_df_3_short.iloc[:, :] = 0
for i in range(len(weight_df_3)):
    for j in range(len(weight_df_3.columns)):

        if weight_df_3_1.iloc[i, j] < 0 and weight_df_3_2.iloc[i, j] <= 0:
            weight_df_3_short.iloc[i, j] = -1
        else:
            weight_df_3_short.iloc[i, j] = 0

ret_df_3_short = open_price.shift(-1).div(vwap_afternoon).sub(1)

nv_df_3_short = weight_df_3_short.mul(ret_df_3_short).fillna(0).cumsum()
pnl_df_3_short = weight_df_3_short.mul(ret_df_3_short)

a = process_pnl_annually(pnl_df_3_short)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_short_{roll_method}.csv")

nv_df_3_short.to_csv(f"{file_save_path}nv_df_3_short_{roll_method}.csv")

weight_df_3_1 = morning_price.div(open_price).sub(1).applymap(np.sign)
weight_df_3_2 = afternoon_price.div(open_price).sub(1).applymap(np.sign)
weight_df_3_long = weight_df_3_1.copy()
weight_df_3_long.iloc[:, :] = 0
for i in range(len(weight_df_3_long)):
    for j in range(len(weight_df_3_long.columns)):

        if weight_df_3_1.iloc[i, j] > 0 and weight_df_3_2.iloc[i, j] > 0:
            weight_df_3_long.iloc[i, j] = 1
        else:
            weight_df_3_long.iloc[i, j] = 0

ret_df_3_long = open_price.shift(-1).div(vwap_afternoon).sub(1)

nv_df_3_long = weight_df_3_long.mul(ret_df_3_long).fillna(0).cumsum()
pnl_df_3_long = weight_df_3_long.mul(ret_df_3_long)

a = process_pnl_annually(pnl_df_3_long)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_long_{roll_method}.csv")

nv_df_3_long.to_csv(f"{file_save_path}nv_df_3_long_{roll_method}.csv")

# signal 4

weight_df_4_2 = afternoon_price.div(open_price).sub(1).applymap(np.sign)
weight_df_4 = weight_df_4_2.copy()
weight_df_4.iloc[:, :] = 0
weight_df_4_1 = weight_df_4_2.copy()
weight_df_4_1.iloc[:, :] = 0

for i in range(len(weight_df_3)):
    for j in range(len(weight_df_3.columns)):
        if high_bar_3.iloc[i, j] <= high_bar_2.iloc[i, j] <= high_bar_1.iloc[i, j]:
            weight_df_4_1.iloc[i, j] = -1
        else:
            if low_bar_1.iloc[i, j] <= low_bar_2.iloc[i, j] <= low_bar_3.iloc[i, j]:
                weight_df_4_1.iloc[i, j] = 1
            else:
                weight_df_4_1.iloc[i, j] = -1

        if weight_df_4_1.iloc[i, j] > 0 and weight_df_4_2.iloc[i, j] >= 0:
            weight_df_4.iloc[i, j] = 1
        elif weight_df_4_1.iloc[i, j] < 0 and weight_df_4_2.iloc[i, j] <= 0:
            weight_df_4.iloc[i, j] = -1
        else:
            weight_df_4.iloc[i, j] = 0

ret_df_4 = open_price.shift(-1).div(vwap_afternoon).sub(1)

nv_df_4 = weight_df_4.mul(ret_df_4).fillna(0).cumsum()
pnl_df_4 = weight_df_4.mul(ret_df_4)

a = process_pnl_annually(pnl_df_4)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_4_{roll_method}.csv")

nv_df_4.to_csv(f"{file_save_path}nv_df_4_{roll_method}.csv")


def ts_joint(price_df_list, price_type_list):
    if len(set([x.shape for x in price_df_list])) == 1:
        joint_price_df = pd.DataFrame(index=range(len(price_df_list[0]) * len(price_df_list)),
                                      columns=list(price_df_list[0].columns) + ['Type', 'Date'])
        for i in range(len(price_df_list[0])):
            for j in range(len(price_df_list)):
                joint_price_df.iloc[i * len(price_df_list) + j, :len(price_df_list)] = price_df_list[j].iloc[i, :]
                joint_price_df.loc[i * len(price_df_list) + j, 'Date'] = price_df_list[j].index[i]
                joint_price_df.loc[i * len(price_df_list) + j, 'Type'] = price_type_list[j]

    else:
        raise ValueError("All dataframe in list should be the same size")
    return joint_price_df


const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_after_fee2_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_after_fee2_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 0 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_after_fee_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_after_fee_{roll_method}.csv")

#
# a=price_df[price_df['Type']=='trade'].iloc[:,:3].reset_index().melt(['Date']).dropna()
# a.columns=['Date','Tickers','Price']
# b=weight_df_3.reset_index().melt(['Date']).replace(0,np.nan).dropna()
# b.columns=['Date','Tickers','Volume']
# trade_df=pd.merge(left=a,right=b,on=['Date','Tickers'],how='outer').dropna()
#
#

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_short)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_short.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_short.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_short_after_fee2_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_short_after_fee2_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 0 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_short)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_short.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_short.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_short_after_fee_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_short_after_fee_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_long)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_long.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_long.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_long_after_fee2_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_long_fee2_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 0 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_long)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_long.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_long.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_long_after_fee_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_long_after_fee_{roll_method}.csv")

###########################
# signal 3 threshold 0.0001
threshold_long = 0.001
threshold_short = -0.001
weight_df_3_1 = morning_price.div(open_price).sub(1)
weight_df_3_2 = afternoon_price.div(open_price).sub(1)
weight_df_3_threshold = weight_df_3_1.copy()
weight_df_3_threshold.iloc[:, :] = 0
for i in range(len(weight_df_3_threshold)):
    for j in range(len(weight_df_3_threshold.columns)):
        if weight_df_3_1.iloc[i, j] > 0 and weight_df_3_2.iloc[i, j] > 0 and weight_df_3_1.iloc[i, j] + \
                weight_df_3_2.iloc[i, j] > threshold_long:
            weight_df_3_threshold.iloc[i, j] = 1
        elif weight_df_3_1.iloc[i, j] < 0 and weight_df_3_2.iloc[i, j] < 0 and weight_df_3_1.iloc[i, j] + \
                weight_df_3_2.iloc[i, j] < threshold_short:
            weight_df_3_threshold.iloc[i, j] = -1
        else:
            weight_df_3_threshold.iloc[i, j] = 0

ret_df_3_threshold = open_price.shift(-1).div(vwap_afternoon).sub(1)

nv_df_3_threshold = weight_df_3_threshold.mul(ret_df_3_threshold).fillna(0).cumsum()
pnl_df_3_threshold = weight_df_3_threshold.mul(ret_df_3_threshold)

a = process_pnl_annually(pnl_df_3_threshold)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"{file_save_path}indicator_df_3_threshold_up_{threshold_long}_down_{threshold_short}_{roll_method}.csv")

nv_df_3_threshold.to_csv(
    f"{file_save_path}nv_df_3_threshold_up_{threshold_long}_down_{threshold_short}_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_threshold)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"{file_save_path}indicator_df_3_threshold_up_{threshold_long}_down_{threshold_short}_{roll_method}_after_fee2.csv")
bt_nv.to_csv(
    f"{file_save_path}nv_df_3_threshold_up_{threshold_long}_down_{threshold_short}_{roll_method}_after_fee2.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 0 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_threshold)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"{file_save_path}indicator_df_3_threshold_up_{threshold_long}_down_{threshold_short}_{roll_method}_after_fee1.csv")
bt_nv.to_csv(
    f"{file_save_path}nv_df_3_threshold_up_{threshold_long}_down_{threshold_short}_{roll_method}_after_fee1.csv")

########################
# signal 5 reverse

# signal 3 threshold 0.0001
weight_df_5 = morning_price.div(open_price).sub(1).applymap(np.sign).mul(-1)
ret_df_5 = vwap_afternoon.div(vwap_morning).sub(1)

nv_df_5 = weight_df_5.mul(ret_df_5).fillna(0).cumsum()
pnl_df_5 = weight_df_5.mul(ret_df_5)
a = process_pnl_annually(pnl_df_5)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_5_reverse_{roll_method}.csv")

nv_df_5.to_csv(f"{file_save_path}nv_df_5_reverse_{roll_method}.csv")

pnl_df_5_fee = weight_df_5.copy()
pnl_df_5_fee.iloc[:, :] = 0
for i in range(len(pnl_df_5_fee)):
    for j in range(len(pnl_df_5_fee.columns)):
        if weight_df_5.iloc[i, j] > 0:
            pnl_df_5_fee.iloc[i, j] = (vwap_afternoon.iloc[i, j] / vwap_morning.iloc[i, j] * (1 - price_impact_rate) / (
                    1 + price_impact_rate) - 1) * weight_df_5.iloc[i, j]
        elif weight_df_5.iloc[i, j] < 0:
            pnl_df_5_fee.iloc[i, j] = (vwap_afternoon.iloc[i, j] / vwap_morning.iloc[i, j] * (1 + price_impact_rate) / (
                    1 - price_impact_rate) - 1) * weight_df_5.iloc[i, j]
        else:
            pnl_df_5_fee.iloc[i, j] = 0
nv_df_5_fee = pnl_df_5_fee.cumsum()
a = process_pnl_annually(pnl_df_5_fee)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_5_reverse_fee_{roll_method}.csv")

nv_df_5_fee.to_csv(f"{file_save_path}nv_df_5_reverse_fee_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [vwap_morning, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_5)):
    weight_df.iloc[i * 3, :len(price_df_list)] = weight_df_5.iloc[i, :].fillna(0)
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = 0

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            # if np.isnan(price_df.iloc[i, j]):
            #     bt_pnl.iloc[i, j] = np.nan
            # else:
            bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate) - price_df.iloc[i - 1, j] * (
                        1 + price_impact_rate)
            else:
                bt_pnl.iloc[i, j] = -price_df.iloc[i, j] * (1 + price_impact_rate) + price_df.iloc[i - 1, j] * (
                        1 - price_impact_rate)
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_5_reverse_after_fee2_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_5_reverse_fee2_{roll_method}.csv")

################################
# signal 3 rolling threshold

# signal 3
weight_df_3_1 = morning_price.div(open_price).sub(1)
weight_df_3_2 = afternoon_price.div(open_price).sub(1)
weight_df_3 = weight_df_3_1.copy()
weight_df_3.iloc[:, :] = 0
for i in range(len(weight_df_3)):
    for j in range(len(weight_df_3.columns)):
        if weight_df_3_1.iloc[i, j] > 0 and weight_df_3_2.iloc[i, j] >= 0:
            weight_df_3.iloc[i, j] = weight_df_3_1.iloc[i, j] + weight_df_3_2.iloc[i, j]
        elif weight_df_3_1.iloc[i, j] < 0 and weight_df_3_2.iloc[i, j] <= 0:
            weight_df_3.iloc[i, j] = weight_df_3_1.iloc[i, j] + weight_df_3_2.iloc[i, j]
        else:
            weight_df_3.iloc[i, j] = np.nan

ret_df_3 = open_price.shift(-1).div(vwap_afternoon).sub(1)


def piece_wise_reg_signal_graph(x, y):
    def piecewise_linear(x, x1, x2, y0, k1, k2):
        return np.piecewise(x, [x < x1, np.logical_and(x >= x1, x < x2), x >= x2],
                            [lambda x: k1 * x + y0 - k1 * x1, lambda x: x * 0 + y0, lambda x: k2 * x + y0 - k2 * x2])

    param, _ = optimize.curve_fit(piecewise_linear, x, y,
                                  bounds=([np.min(x), 0, -np.inf, -np.inf, 0], [0, np.max(x), np.inf, 0, np.inf]))
    return param


long_threshold = weight_df_3.copy()
short_threshold = weight_df_3.copy()
for j in range(len(weight_df_3.columns)):
    for i in range(len(weight_df_3)):
        if i < 100:
            long_threshold.iloc[i, j] = 0
            short_threshold.iloc[i, j] = 0
        else:
            df = pd.concat([weight_df_3.iloc[:i, j], ret_df_3.iloc[:i, j]], axis=1)
            df.columns = ['signal', 'ret']
            df = df.dropna()
            if len(df) < 100:
                long_threshold.iloc[i, j] = 0
                short_threshold.iloc[i, j] = 0
            else:
                try:
                    df = df.sort_values(['signal'])
                    df['cumpnl'] = df['ret'].cumsum()
                    x = np.array(df['signal'])
                    y = np.array(df['cumpnl'])
                    piecewise_result = piece_wise_reg_signal_graph(x, y)
                    long_threshold.iloc[i, j] = piecewise_result[0]
                    short_threshold.iloc[i, j] = piecewise_result[1]
                except:
                    long_threshold.iloc[i, j] = 0
                    short_threshold.iloc[i, j] = 0
        print(f"{i}/{len(weight_df_3)} {j} is ok")

long_threshold_roll_mean = long_threshold.rolling(100).mean()
short_threshold_roll_mean = short_threshold.rolling(100).mean()

weight_df_3_piecewise_reg = weight_df_3.copy()
for i in range(len(weight_df_3_piecewise_reg)):
    if i == 0:
        weight_df_3_piecewise_reg.iloc[i, :] = 0
    else:
        for j in range(len(weight_df_3_piecewise_reg.columns)):
            if weight_df_3.iloc[i, j] > 0:
                if weight_df_3.iloc[i, j] > short_threshold.iloc[i - 1, j]:
                    weight_df_3_piecewise_reg.iloc[i, j] = 1
                else:
                    weight_df_3_piecewise_reg.iloc[i, j] = 0
            elif weight_df_3.iloc[i, j] > 0:
                if weight_df_3.iloc[i, j] < long_threshold.iloc[i - 1, j]:
                    weight_df_3_piecewise_reg.iloc[i, j] = -1
                else:
                    weight_df_3_piecewise_reg.iloc[i, j] = 0
            else:
                weight_df_3_piecewise_reg.iloc[i, j] = 0

nv_df_3_piecewise_reg = weight_df_3_piecewise_reg.mul(ret_df_3).fillna(0).cumsum()
pnl_df_3_piecewise_reg = weight_df_3_piecewise_reg.mul(ret_df_3)

a = process_pnl_annually(pnl_df_3_piecewise_reg)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_piecewise_reg_{roll_method}.csv")

nv_df_3_piecewise_reg.to_csv(f"{file_save_path}nv_df_3_piecewise_reg_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_piecewise_reg)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_piecewise_reg.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_piecewise_reg.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_piecewise_reg_after_fee2_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_piecewise_reg_after_fee2_{roll_method}.csv")

# piecewise rolling mean

weight_df_3_piecewise_roll_mean_reg = weight_df_3.copy()
for i in range(len(weight_df_3_piecewise_roll_mean_reg)):
    if i == 0:
        weight_df_3_piecewise_roll_mean_reg.iloc[i, :] = 0
    else:
        for j in range(len(weight_df_3_piecewise_roll_mean_reg.columns)):
            if weight_df_3.iloc[i, j] > 0:
                if weight_df_3.iloc[i, j] > short_threshold_roll_mean.iloc[i - 1, j]:
                    weight_df_3_piecewise_roll_mean_reg.iloc[i, j] = 1
                else:
                    weight_df_3_piecewise_roll_mean_reg.iloc[i, j] = 0
            elif weight_df_3.iloc[i, j] > 0:
                if weight_df_3.iloc[i, j] < long_threshold_roll_mean.iloc[i - 1, j]:
                    weight_df_3_piecewise_roll_mean_reg.iloc[i, j] = -1
                else:
                    weight_df_3_piecewise_roll_mean_reg.iloc[i, j] = 0
            else:
                weight_df_3_piecewise_roll_mean_reg.iloc[i, j] = 0

nv_df_3_piecewise_reg_roll_mean = weight_df_3_piecewise_roll_mean_reg.mul(ret_df_3).fillna(0).cumsum()
pnl_df_3_piecewise_reg_roll_mean = weight_df_3_piecewise_roll_mean_reg.mul(ret_df_3)

a = process_pnl_annually(pnl_df_3_piecewise_reg_roll_mean)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_piecewise_reg_roll_mean_{roll_method}.csv")

nv_df_3_piecewise_reg_roll_mean.to_csv(f"{file_save_path}nv_df_3_piecewise_reg_roll_mean_{roll_method}.csv")

const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [open_price, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(weight_df_3_piecewise_roll_mean_reg)):
    weight_df.iloc[i * 3, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_piecewise_roll_mean_reg.iloc[i, :]
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_piecewise_roll_mean_reg.iloc[i, :]

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_3_piecewise_reg_roll_mean_after_fee2_{roll_method}.csv")
bt_nv.to_csv(f"{file_save_path}nv_df_3_piecewise_reg_roll_mean_after_fee2_{roll_method}.csv")

############################
# reverse
volume_daily = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['vol'], freq='Daily', index=1, ret_index=False,
                                                           roll_method=roll_method)

volume_daily['Fut_code'] = volume_daily['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume_daily = volume_daily.pivot_table(index='Trade_DT', columns='Fut_code', values=volume_daily.columns[2])
adv = volume_daily.rolling(5).mean()


def get_ts_volume(volume_minute_df, datetime_time_list):
    volume_all = pd.DataFrame()
    for datetime_minute in datetime_time_list:
        volume_one_time = volume_minute_df[volume_minute_df['Time'] == datetime_minute][['Fut_code', 'Date', 'volume']]

        if len(volume_all) == 0:
            volume_all = volume_one_time
        else:
            volume_all = pd.merge(left=volume_all, right=volume_one_time, on=['Fut_code', 'Date'])

    volume_all['lowest'] = volume_all.iloc[:, 2:].sum(axis=1)
    volume_all = volume_all.pivot_table(index='Date', columns='Fut_code', values='lowest')
    return volume_all


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
# morning vwap 15th-20th
vwap_morning_part_1 = get_ts_volume(volume_minute,
                                    [datetime.time(9, 16), datetime.time(9, 17), datetime.time(9, 18),
                                     datetime.time(9, 19), datetime.time(9, 20)])
vwap_morning_part_1 = vwap_morning_part_1[vwap_morning_part_1.index < change_trading_time_start_date]

vwap_morning_part_2 = get_ts_volume(volume_minute,
                                    [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                     datetime.time(9, 34), datetime.time(9, 35)])
vwap_morning_part_2 = vwap_morning_part_2[vwap_morning_part_2.index >= change_trading_time_start_date]
vwap_morning_part_2 = vwap_morning_part_2[vwap_morning_part_2.index < database_change_record_rule_start_date]

vwap_morning_part_3 = get_ts_volume(volume_minute,
                                    [datetime.time(9, 30), datetime.time(9, 31), datetime.time(9, 32),
                                     datetime.time(9, 33), datetime.time(9, 34)])
vwap_morning_part_3 = vwap_morning_part_3[vwap_morning_part_3.index >= database_change_record_rule_start_date]

volume_first_5min = pd.concat([vwap_morning_part_1, vwap_morning_part_2, vwap_morning_part_3])

volume_first_5min_normalize = volume_first_5min / adv.shift(1)

signal_1 = morning_price.div(open_price).sub(1)
signal_1_add_volume = signal_1.copy()
for i in range(len(signal_1_add_volume)):
    for j in range(len(signal_1_add_volume.columns)):
        if signal_1.iloc[i, j] > 0 and volume_first_5min_normalize.iloc[i, j] > 0.5:
            signal_1_add_volume.iloc[i, j] = -1
        elif signal_1.iloc[i, j] < 0 and volume_first_5min_normalize.iloc[i, j] < 0.5:
            signal_1_add_volume.iloc[i, j] = 1
        else:
            signal_1_add_volume.iloc[i, j] = 0
ret_df_morning_vwap_to_afternoon_price = afternoon_price.div(vwap_morning).sub(1)

nv_df_1_add_volume = signal_1_add_volume.mul(ret_df_morning_vwap_to_afternoon_price).fillna(0).cumsum()
pnl_df_1_add_volume = signal_1_add_volume.mul(ret_df_morning_vwap_to_afternoon_price)

a = process_pnl_annually(pnl_df_1_add_volume)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_1_add_volume_{roll_method}.csv")

nv_df_1_add_volume.to_csv(f"{file_save_path}nv_df_1_add_volume_{roll_method}.csv")


def bt(weight_df_3_threshold, ret_df_3_threshold, name_str):
    nv_df_3_threshold = weight_df_3_threshold.mul(ret_df_3_threshold).fillna(0).cumsum()
    pnl_df_3_threshold = weight_df_3_threshold.mul(ret_df_3_threshold)

    a = process_pnl_annually(pnl_df_3_threshold)
    indicator_df = [v for _, v in a.items()]
    indicator_df = pd.concat(indicator_df)
    indicator_df.to_csv(
        f"{file_save_path}_indicator_df_{name_str}.csv")

    nv_df_3_threshold.to_csv(
        f"{file_save_path}nv_df_{name_str}.csv")

    const_fee = 3
    price_impact_rate = 1 / 100 / 100
    price_impact_rate1 = 1 / 100 / 100
    price_df_list = [open_price, vwap_afternoon, close_price]
    price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
    price_df.index = price_df['Date']
    price_df = price_df.iloc[:, :len(price_df_list) + 1]
    price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

    weight_df = price_df.iloc[:, :len(price_df_list)].copy()
    weight_df.iloc[:, :len(price_df_list)] = np.nan
    bt_initial_amount_df = weight_df.copy()
    bt_pnl = bt_initial_amount_df.copy()
    bt_final_amount = bt_initial_amount_df.copy()
    bt_trade_amount = bt_initial_amount_df.copy()

    for i in range(len(weight_df_3_threshold)):
        weight_df.iloc[i * 3, :len(price_df_list)] = 0
        weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]
        weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]

    bt_trade_volume = weight_df.copy()
    bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

    for i in range(len(bt_pnl)):
        if i == 0:
            for j in range(len(bt_pnl.columns)):
                if np.isnan(price_df.iloc[i, j]):
                    bt_pnl.iloc[i, j] = np.nan
                else:
                    bt_pnl.iloc[i, j] = 0
        else:
            for j in range(len(bt_pnl.columns)):
                if bt_trade_volume.iloc[i - 1, j] == 0:
                    bt_pnl.iloc[i, j] = 0
                elif bt_trade_volume.iloc[i - 1, j] > 0:
                    if price_df.iloc[i, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                    else:
                        if price_df.iloc[i - 1, -1] == 'close':
                            bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
                else:
                    if price_df.iloc[i, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                    else:
                        if price_df.iloc[i - 1, -1] == 'close':
                            bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
                bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
    bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
    bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
        [100 * 10000, 100 * 10000, 200 * 10000])

    bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
    ret_nv = bt_nv.diff()
    a = process_pnl_annually(ret_nv)
    indicator_df = [v for _, v in a.items()]
    indicator_df = pd.concat(indicator_df)
    indicator_df.to_csv(
        f"{file_save_path}indicator_df_{name_str}_after_fee2.csv")
    bt_nv.to_csv(
        f"{file_save_path}nv_df_{name_str}_after_fee2.csv")

    const_fee = 3
    price_impact_rate = 1 / 100 / 100
    price_impact_rate1 = 0 / 100 / 100
    price_df_list = [open_price, vwap_afternoon, close_price]
    price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
    price_df.index = price_df['Date']
    price_df = price_df.iloc[:, :len(price_df_list) + 1]
    price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

    weight_df = price_df.iloc[:, :len(price_df_list)].copy()
    weight_df.iloc[:, :len(price_df_list)] = np.nan
    bt_initial_amount_df = weight_df.copy()
    bt_pnl = bt_initial_amount_df.copy()
    bt_final_amount = bt_initial_amount_df.copy()
    bt_trade_amount = bt_initial_amount_df.copy()

    for i in range(len(weight_df_3_threshold)):
        weight_df.iloc[i * 3, :len(price_df_list)] = 0
        weight_df.iloc[i * 3 + 1, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]
        weight_df.iloc[i * 3 + 2, :len(price_df_list)] = weight_df_3_threshold.iloc[i, :]

    bt_trade_volume = weight_df.copy()
    bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

    for i in range(len(bt_pnl)):
        if i == 0:
            for j in range(len(bt_pnl.columns)):
                if np.isnan(price_df.iloc[i, j]):
                    bt_pnl.iloc[i, j] = np.nan
                else:
                    bt_pnl.iloc[i, j] = 0
        else:
            for j in range(len(bt_pnl.columns)):
                if bt_trade_volume.iloc[i - 1, j] == 0:
                    bt_pnl.iloc[i, j] = 0
                elif bt_trade_volume.iloc[i - 1, j] > 0:
                    if price_df.iloc[i, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                    else:
                        if price_df.iloc[i - 1, -1] == 'close':
                            bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
                else:
                    if price_df.iloc[i, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                    else:
                        if price_df.iloc[i - 1, -1] == 'close':
                            bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
                bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
    bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
    bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
        [100 * 10000, 100 * 10000, 200 * 10000])

    bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
    ret_nv = bt_nv.diff()
    a = process_pnl_annually(ret_nv)
    indicator_df = [v for _, v in a.items()]
    indicator_df = pd.concat(indicator_df)
    indicator_df.to_csv(
        f"{file_save_path}indicator_df_{name_str}_after_fee1.csv")
    bt_nv.to_csv(
        f"{file_save_path}nv_df_3_{name_str}_after_fee1.csv")


##################################################
# signal 3
weight_df_3_1 = morning_price.sub(open_price).div(0.005)
weight_df_3_2 = afternoon_price.sub(open_price).div(0.005)


ret_df_3 = open_price.shift(-1).div(vwap_afternoon).sub(1)

for m in range(40):
    threshold_long=m
    threshold_short=-m
    weight_df_3_threshold=weight_df_3_1.copy()
    weight_df_3_threshold.iloc[:, :] = 0
    for i in range(len(weight_df_3_threshold)):
        for j in range(len(weight_df_3_threshold.columns)):
            if weight_df_3_1.iloc[i, j] > 0 and weight_df_3_2.iloc[i, j] > 0 and weight_df_3_1.iloc[i, j] + \
                    weight_df_3_2.iloc[i, j] > threshold_long:
                weight_df_3_threshold.iloc[i, j] = 1
            elif weight_df_3_1.iloc[i, j] < 0 and weight_df_3_2.iloc[i, j] < 0 and weight_df_3_1.iloc[i, j] + \
                    weight_df_3_2.iloc[i, j] < threshold_short:
                weight_df_3_threshold.iloc[i, j] = -1
            else:
                weight_df_3_threshold.iloc[i, j] = 0
    name_str=f"signal_3_long_{threshold_long}_tick_short_{threshold_short}_tick_oi"
    bt(weight_df_3_threshold,ret_df_3,name_str)
    print(f"{m} is ok")

###################################
# signal_7: 15th ret* 15th vol
volume_daily = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=['vol'], freq='Daily', index=1, ret_index=False,
                                                           roll_method=roll_method)

volume_daily['Fut_code'] = volume_daily['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume_daily = volume_daily.pivot_table(index='Trade_DT', columns='Fut_code', values=volume_daily.columns[2])
adv = volume_daily.rolling(5).mean()


def get_ts_volume(volume_minute_df, datetime_time_list):
    volume_all = pd.DataFrame()
    for datetime_minute in datetime_time_list:
        volume_one_time = volume_minute_df[volume_minute_df['Time'] == datetime_minute][['Fut_code', 'Date', 'volume']]

        if len(volume_all) == 0:
            volume_all = volume_one_time
        else:
            volume_all = pd.merge(left=volume_all, right=volume_one_time, on=['Fut_code', 'Date'])

    volume_all['lowest'] = volume_all.iloc[:, 2:].sum(axis=1)
    volume_all = volume_all.pivot_table(index='Date', columns='Fut_code', values='lowest')
    return volume_all


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
# morning vwap 15th-20th
vwap_morning_part_1 = get_ts_volume(volume_minute,
                                    [datetime.time(9, 16), datetime.time(9, 17), datetime.time(9, 18),
                                     datetime.time(9, 19), datetime.time(9, 20)])
vwap_morning_part_1 = vwap_morning_part_1[vwap_morning_part_1.index < change_trading_time_start_date]

vwap_morning_part_2 = get_ts_volume(volume_minute,
                                    [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                     datetime.time(9, 34), datetime.time(9, 35)])
vwap_morning_part_2 = vwap_morning_part_2[vwap_morning_part_2.index >= change_trading_time_start_date]
vwap_morning_part_2 = vwap_morning_part_2[vwap_morning_part_2.index < database_change_record_rule_start_date]

vwap_morning_part_3 = get_ts_volume(volume_minute,
                                    [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33),
                                     datetime.time(9, 34), datetime.time(9, 35)])
vwap_morning_part_3 = vwap_morning_part_3[vwap_morning_part_3.index >= database_change_record_rule_start_date]

volume_first_5min = pd.concat([vwap_morning_part_1, vwap_morning_part_2, vwap_morning_part_3])

volume_first_5min_normalize = volume_first_5min / adv.shift(1)

signal_1=morning_price.sub(open_price).div(0.005)
signal_7=volume_first_5min_normalize.applymap(lambda x:1 if x>0.05 else 0).mul(signal_1).applymap(np.sign).mul(-1)
ret_df_7=vwap_afternoon.div(vwap_morning).sub(1)

nv_df_7 = signal_7.mul(ret_df_7).fillna(0).cumsum()
pnl_df_7 = signal_7.mul(ret_df_7)

a = process_pnl_annually(pnl_df_7)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(f"{file_save_path}indicator_df_7_add_volume_reverse_{roll_method}.csv")

nv_df_7.to_csv(f"{file_save_path}nv_df_7_add_volume_reverse_{roll_method}.csv")

name_str='7_add_volume_reverse_oi'
const_fee = 3
price_impact_rate = 1 / 100 / 100
price_impact_rate1 = 1 / 100 / 100
price_df_list = [vwap_morning, vwap_afternoon, close_price]
price_df = ts_joint(price_df_list, ['trade', 'trade', 'close'])
price_df.index = price_df['Date']
price_df = price_df.iloc[:, :len(price_df_list) + 1]
price_df.iloc[:, :len(price_df_list)] = price_df.iloc[:, :len(price_df_list)].mul(bpv_value)

weight_df = price_df.iloc[:, :len(price_df_list)].copy()
weight_df.iloc[:, :len(price_df_list)] = np.nan
bt_initial_amount_df = weight_df.copy()
bt_pnl = bt_initial_amount_df.copy()
bt_final_amount = bt_initial_amount_df.copy()
bt_trade_amount = bt_initial_amount_df.copy()

for i in range(len(signal_7)):
    weight_df.iloc[i * 3, :len(price_df_list)] = signal_7.iloc[i, :]
    weight_df.iloc[i * 3 + 1, :len(price_df_list)] = 0
    weight_df.iloc[i * 3 + 2, :len(price_df_list)] = 0

bt_trade_volume = weight_df.copy()
bt_const_cost_fee = bt_trade_volume.diff().abs().fillna(0).mul(const_fee)

for i in range(len(bt_pnl)):
    if i == 0:
        for j in range(len(bt_pnl.columns)):
            if np.isnan(price_df.iloc[i, j]):
                bt_pnl.iloc[i, j] = np.nan
            else:
                bt_pnl.iloc[i, j] = 0
    else:
        for j in range(len(bt_pnl.columns)):
            if bt_trade_volume.iloc[i - 1, j] == 0:
                bt_pnl.iloc[i, j] = 0
            elif bt_trade_volume.iloc[i - 1, j] > 0:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 + price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 - price_impact_rate1) - price_df.iloc[i - 1, j]
            else:
                if price_df.iloc[i, -1] == 'close':
                    bt_pnl.iloc[i, j] = price_df.iloc[i, j] - price_df.iloc[i - 1, j] * (1 - price_impact_rate)
                else:
                    if price_df.iloc[i - 1, -1] == 'close':
                        bt_pnl.iloc[i, j] = price_df.iloc[i, j] * (1 + price_impact_rate1) - price_df.iloc[i - 1, j]
            bt_pnl.iloc[i, j] = bt_pnl.iloc[i, j] * bt_trade_volume.iloc[i - 1, j]
bt_cumpnl = (bt_pnl - bt_const_cost_fee).cumsum()
bt_cumpnl = bt_cumpnl.fillna(method='ffill').fillna(0).add([100 * 10000, 100 * 10000, 200 * 10000]).div(
    [100 * 10000, 100 * 10000, 200 * 10000])

bt_nv = bt_cumpnl.groupby(bt_cumpnl.index)[bt_cumpnl.columns].last()
ret_nv = bt_nv.diff()
a = process_pnl_annually(ret_nv)
indicator_df = [v for _, v in a.items()]
indicator_df = pd.concat(indicator_df)
indicator_df.to_csv(
    f"{file_save_path}indicator_df_{name_str}_after_fee2.csv")
bt_nv.to_csv(
    f"{file_save_path}nv_df_{name_str}_after_fee2.csv")

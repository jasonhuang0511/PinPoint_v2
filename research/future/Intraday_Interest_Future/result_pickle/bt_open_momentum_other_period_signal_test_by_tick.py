import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

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


file_save_path = "C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\"
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
high_minute_part_3 = get_ts_high(high_minute_price, [datetime.time(9, 30), datetime.time(9, 31), datetime.time(9, 32),
                                                     datetime.time(9, 33), datetime.time(9, 34)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
high_bar_1 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_high(high_minute_price, [datetime.time(9, 21), datetime.time(9, 22), datetime.time(9, 23),
                                                     datetime.time(9, 24), datetime.time(9, 25)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_high(high_minute_price, [datetime.time(9, 36), datetime.time(9, 37), datetime.time(9, 38),
                                                     datetime.time(9, 39), datetime.time(9, 40)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_high(high_minute_price, [datetime.time(9, 35), datetime.time(9, 36), datetime.time(9, 37),
                                                     datetime.time(9, 38), datetime.time(9, 39)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
high_bar_2 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_high(high_minute_price, [datetime.time(9, 26), datetime.time(9, 27), datetime.time(9, 28),
                                                     datetime.time(9, 29), datetime.time(9, 30)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_high(high_minute_price, [datetime.time(9, 41), datetime.time(9, 42), datetime.time(9, 43),
                                                     datetime.time(9, 44), datetime.time(9, 45)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_high(high_minute_price, [datetime.time(9, 40), datetime.time(9, 41), datetime.time(9, 42),
                                                     datetime.time(9, 43), datetime.time(9, 44)])
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
high_minute_part_3 = get_ts_low(high_minute_price, [datetime.time(9, 30), datetime.time(9, 31), datetime.time(9, 32),
                                                    datetime.time(9, 33), datetime.time(9, 34)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
low_bar_1 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_low(high_minute_price, [datetime.time(9, 21), datetime.time(9, 22), datetime.time(9, 23),
                                                    datetime.time(9, 24), datetime.time(9, 25)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_low(high_minute_price, [datetime.time(9, 36), datetime.time(9, 37), datetime.time(9, 38),
                                                    datetime.time(9, 39), datetime.time(9, 40)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_low(high_minute_price, [datetime.time(9, 35), datetime.time(9, 36), datetime.time(9, 37),
                                                    datetime.time(9, 38), datetime.time(9, 39)])
high_minute_part_3 = high_minute_part_3[high_minute_part_3.index >= database_change_record_rule_start_date]
low_bar_2 = pd.concat([high_minute_part_1, high_minute_part_2, high_minute_part_3])

high_minute_part_1 = get_ts_low(high_minute_price, [datetime.time(9, 26), datetime.time(9, 27), datetime.time(9, 28),
                                                    datetime.time(9, 29), datetime.time(9, 30)])
high_minute_part_1 = high_minute_part_1[high_minute_part_1.index < change_trading_time_start_date]
high_minute_part_2 = get_ts_low(high_minute_price, [datetime.time(9, 41), datetime.time(9, 42), datetime.time(9, 43),
                                                    datetime.time(9, 44), datetime.time(9, 45)])
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index >= change_trading_time_start_date]
high_minute_part_2 = high_minute_part_2[high_minute_part_2.index < database_change_record_rule_start_date]
high_minute_part_3 = get_ts_low(high_minute_price, [datetime.time(9, 40), datetime.time(9, 41), datetime.time(9, 42),
                                                    datetime.time(9, 43), datetime.time(9, 44)])
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

morning_price_part3 = close_minute_price[close_minute_price['Time'] == datetime.time(9, 44)]
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
                                  [datetime.time(9, 45), datetime.time(9, 46), datetime.time(9, 47),
                                   datetime.time(9, 48), datetime.time(9, 49)])
vwap_morning_part_3 = vwap_morning_part_3[vwap_morning_part_3.index >= database_change_record_rule_start_date]

vwap_morning = pd.concat([vwap_morning_part_1, vwap_morning_part_2, vwap_morning_part_3])

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


from backtest.single_factor_signal_graph import single_factor_multi_asset_signal_graph

signal_graph_save_path = "C:\\Users\\jason.huang\\PycharmProjects\\PinPoint_v2\\research\\future\\Intraday_Interest_Future\\result_pickle\\signal_graph\\tick\\"
tick_basis=0.005

ret_df_1 = vwap_close.div(vwap_morning).sub(1)
ret_df_2 = vwap_close.div(vwap_afternoon).sub(1)
ret_df_3 = open_price.shift(-1).div(vwap_afternoon).sub(1)
ret_df_overnight = open_price.shift(-1).div(close_price).sub(1)
ret_df_morning_vwap_to_afternoon_price=afternoon_price.div(vwap_morning).sub(1)

# signal_15th_ret_vs_ret_15th_20th_trade_in_1510_1515_trade_out
signal_1 = morning_price.sub(open_price).div(tick_basis)
save_location = f"{signal_graph_save_path}signal_15th_ret_vs_ret_15th_20th_trade_in_1510_1515_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_1, ret_df=ret_df_1, save_location=save_location)

# signal_15th_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_1, ret_df=ret_df_2, save_location=save_location)

# signal_15th_ret_vs_ret_1330_1335_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_vs_ret_1330_1335_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_1, ret_df=ret_df_3, save_location=save_location)

# signal_15th_ret_vs_ret_close_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_vs_ret_close_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_1, ret_df=ret_df_overnight, save_location=save_location)

# signal_15th_ret_vs_ret_15th_20th_in_1330_1335_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_vs_ret_15th_20th_in_1330_1335_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_1, ret_df=ret_df_morning_vwap_to_afternoon_price, save_location=save_location)


################################
signal_2 = afternoon_price.sub(open_price).div(tick_basis)
# signal_open_1330_ret_vs_ret_1330_1335_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_open_1330_ret_vs_ret_1330_1335_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_2, ret_df=ret_df_3, save_location=save_location)

# signal_open_1330_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out
save_location = f"{signal_graph_save_path}signal_open_1330_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_2, ret_df=ret_df_2, save_location=save_location)

# signal_open_1330_ret_vs_ret_close_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_open_1330_ret_vs_ret_close_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_2, ret_df=ret_df_overnight, save_location=save_location)

############################################
signal_3 = signal_2.copy()
for i in range(len(signal_3)):
    for j in range(len(signal_3.columns)):
        if signal_1.iloc[i, j] > 0 and signal_2.iloc[i, j] > 0:
            signal_3.iloc[i, j] = signal_1.iloc[i, j] + signal_2.iloc[i, j]
        elif signal_1.iloc[i, j] < 0 and signal_2.iloc[i, j] < 0:
            signal_3.iloc[i, j] = signal_1.iloc[i, j] + signal_2.iloc[i, j]
        else:
            signal_3.iloc[i, j] = np.nan
# signal_15th_ret_&_open_1330_ret_vs_ret_1330_1335_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_&_open_1330_ret_vs_ret_1330_1335_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_3, ret_df=ret_df_3, save_location=save_location)

# signal_15th_ret_&_open_1330_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_&_open_1330_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_3, ret_df=ret_df_2, save_location=save_location)

# signal_15th_ret_&_open_1330_ret_vs_ret_close_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_&_open_1330_ret_vs_ret_close_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_3, ret_df=ret_df_overnight, save_location=save_location)


signal_4 = signal_1.add(signal_2)
# signal_15th_ret_add_open_1330_ret_vs_ret_1330_1335_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_add_open_1330_ret_vs_ret_1330_1335_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_4, ret_df=ret_df_3, save_location=save_location)

# signal_15th_ret_add_open_1330_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_add_open_1330_ret_vs_ret_1330_1335_trade_in_1510_1515_trade_out{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_4, ret_df=ret_df_2, save_location=save_location)

# signal_15th_ret_add_open_1330_ret_vs_ret_close_trade_in_next_open_trade_out
save_location = f"{signal_graph_save_path}signal_15th_ret_add_open_1330_ret_vs_ret_close_trade_in_next_open_trade_out_{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_4, ret_df=ret_df_overnight, save_location=save_location)

####################################################
# research
volume_daily=ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                         end_date=end_date,
                                                         key_word=['vol'], freq='Daily', index=1, ret_index=False,
                                                         roll_method=roll_method)

volume_daily['Fut_code'] = volume_daily['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume_daily = volume_daily.pivot_table(index='Trade_DT', columns='Fut_code', values=volume_daily.columns[2])
adv=volume_daily.rolling(5).mean()



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


volume_first_5min_normalize=volume_first_5min/adv.shift(1)
signal_6 = volume_first_5min_normalize
# signal_15th_volume_adjust_vs_ret_15th_20th_in_1330_1335_trade_out
save_location = f"{signal_graph_save_path}signal_15th_volume_adjust_vs_ret_15th_20th_in_1330_1335_trade_out{roll_method}.jpeg"
single_factor_multi_asset_signal_graph(signal_df=signal_6, ret_df=ret_df_morning_vwap_to_afternoon_price, save_location=save_location)











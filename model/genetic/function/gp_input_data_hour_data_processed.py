import os.path
import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import model.constants.futures as ConstFutBasic
import model.constants.path as ConstPath

import time

warnings.filterwarnings('ignore')
import multiprocessing


def process_daily_min_data(df, key, method, start_date, tickers_list=ConstFutBasic.fut_code_list):
    end_date_hour_list = {k: [i[1] for i in v if i[1] != '10:15'] for k, v in
                          ConstFutBasic.fut_code_trading_min_time_dict.items()}
    start_date_hour_list = {k: [i[0] for i in v if i[0] != '10:30'] for k, v in
                            ConstFutBasic.fut_code_trading_min_time_dict.items()}

    mapping_start_date_to_result_min_data_dict = {'21:00': '08:00', "09:00": "12:00", "09:30": "12:00",
                                                  "09:15": "12:00", "13:30": "16:00", "13:00": "16:00"}

    df['Fut_code'] = df['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
    df['Time'] = [str(x)[11:16] if int(str(x)[11:13]) > 8 else str(int(str(x)[11:13]) + 24) + str(x)[13:16] for x in
                  df['Trade_DT']]
    data_all = pd.DataFrame()

    for tickers in tickers_list:

        df_selected = df[df['Fut_code'] == tickers]
        df_selected = df_selected.sort_values(['Trade_DT'])
        df_selected = df_selected.reset_index(drop=True)
        start_date_hour_time = start_date_hour_list[tickers]
        end_date_hour_time = end_date_hour_list[tickers]
        if tickers == 'T.CFE' or tickers == 'TF.CFE':
            start_date_hour_time[0] = '09:15'

        if int(str(end_date_hour_time[0])[:2]) < 8:
            end_date_hour_time[0] = str(int(str(end_date_hour_time[0])[:2]) + 24) + str(end_date_hour_time[0])[2:]

        data = pd.DataFrame()
        if len(df_selected) > 0:
            for i in range(len(start_date_hour_time)):
                start_date_hour = start_date_hour_time[i]
                end_date_hour = end_date_hour_time[i]
                df_selected.loc[(df_selected['Time'] >= start_date_hour) & (df_selected[
                                                                                'Time'] <= end_date_hour), 'Index'] = start_date + ' ' + \
                                                                                                                      mapping_start_date_to_result_min_data_dict[
                                                                                                                          start_date_hour] + ':00'
            data = eval(f"df_selected.groupby('Index')[\'{key}\'].{method}()").reset_index()
            data['Fut_code'] = tickers
            data.columns = ['Trade_DT', key, 'Fut_code']
            data = data[['Fut_code', 'Trade_DT', key]]
            data_all = pd.concat([data_all, data], axis=0)
        else:
            pass
    data_all = data_all.reset_index(drop=True)
    return data_all


def semi_daily_data_save(date, tickers_list=ConstFutBasic.fut_code_list):
    print(f"{date} starts")
    save_path = 'C:\\Users\\jason.huang\\research\\data_mining\\GP\\features\\data_half_daily_loop_one\\' + date + '\\'
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except:
            os.makedirs(save_path)
    start_date = date
    end_date = date
    a = time.time()
    close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                              key_word=['close'], freq='min', index=1, ret_index=False)
    twap = process_daily_min_data(close, "close", "mean", start_date=start_date, tickers_list=tickers_list)
    twap.to_csv(save_path + 'freq_4H_twap.csv')
    print('twap is ok')

    close = process_daily_min_data(close, "close", "last", start_date=start_date, tickers_list=tickers_list)
    close.to_csv(save_path + 'freq_4H_close.csv')
    print('close is ok')

    open = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                             key_word=['open'], freq='min', index=1, ret_index=False)
    open = process_daily_min_data(open, "open", "first", start_date=start_date, tickers_list=tickers_list)
    open.to_csv(save_path + 'freq_4H_open.csv')
    print('open is ok')

    high = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                             key_word=['high'], freq='min', index=1, ret_index=False)
    high = process_daily_min_data(high, "high", "max", start_date=start_date, tickers_list=tickers_list)
    high.to_csv(save_path + 'freq_4H_high.csv')
    print('high is ok')

    low = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                            key_word=['low'], freq='min', index=1, ret_index=False)
    low = process_daily_min_data(low, "low", "min", start_date=start_date, tickers_list=tickers_list)
    low.to_csv(save_path + 'freq_4H_low.csv')
    print('low is ok')

    volume = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                               key_word=['volume'], freq='min', index=1, ret_index=False)
    volume = process_daily_min_data(volume, "volume", "sum", start_date=start_date, tickers_list=tickers_list)
    volume.to_csv(save_path + 'freq_4H_volume.csv')
    print('volume is ok')

    amount = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                               key_word=['amount'], freq='min', index=1, ret_index=False)
    amount = process_daily_min_data(amount, "amount", "sum", start_date=start_date, tickers_list=tickers_list)
    amount.to_csv(save_path + 'freq_4H_amount.csv')
    print('amount is ok')

    position = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                 key_word=['position'], freq='min', index=1, ret_index=False)
    position_max = process_daily_min_data(position, "position", "max", start_date=start_date, tickers_list=tickers_list)
    position_min = process_daily_min_data(position, "position", "min", start_date=start_date, tickers_list=tickers_list)
    position_range = pd.merge(left=position_max, right=position_min, how='inner', on=['Fut_code', 'Trade_DT'])
    position_range['position_range'] = position_range['position_x'] - position_range['position_y']
    position_range = position_range[['Fut_code', 'Trade_DT', 'position_range']]
    position_range.to_csv(save_path + 'freq_4H_position_range.csv')
    print('position_range is ok')

    position = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                 key_word=['position'], freq='min', index=1, ret_index=False)
    position = process_daily_min_data(position, "position", "last", start_date=start_date, tickers_list=tickers_list)
    position.to_csv(save_path + 'freq_4H_position.csv')
    print('position is ok')

    volume1 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                key_word=['volume'], freq='min', index=1, ret_index=False)
    amount1 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                key_word=['amount'], freq='min', index=1, ret_index=False)
    vwap = pd.merge(left=amount1, right=volume1, how='inner', on=['Code', 'Trade_DT'])
    vwap['t_vwap'] = vwap['amount'] / vwap['volume']

    t_vwap = process_daily_min_data(vwap, "t_vwap", "mean", start_date=start_date, tickers_list=tickers_list)
    t_vwap.to_csv(save_path + 'freq_4H_t_vwap.csv')
    print('t_vwap is ok')

    print(f"total time: {time.time() - a}")
    print(f"{date} is ok")


def period_daily_loop_semi_daily_data():
    tickers_list = ConstFutBasic.fut_code_list
    start_date_period = datetime.date(2018, 1, 1)
    end_date_period = datetime.date(2022, 9, 30)
    trading_canlendar = np.array(ExtractDataPostgre.get_trading_calendar().values)
    trading_canlendar = trading_canlendar[trading_canlendar >= start_date_period]
    trading_canlendar = trading_canlendar[trading_canlendar <= end_date_period]
    trading_canlendar = [datetime.datetime.strftime(x, "%Y-%m-%d") for x in trading_canlendar]
    trading_canlendar = trading_canlendar[::-1]
    for date in trading_canlendar:
        semi_daily_data_save(date, tickers_list)


def read_file_location(date, file_name="freq_4H_amount"):
    save_path = 'C:\\Users\\jason.huang\\research\\data_mining\\GP\\features\\data_half_daily\\'
    file_location = save_path + date + '\\' + file_name + '.csv'
    data = pd.read_csv(file_location)
    return data


def joint_a_table(file_path='C:\\Users\\jason.huang\\research\\data_mining\\GP\\features\\data_half_daily_loop_one\\'):
    data_all = pd.DataFrame()
    for date_name in os.listdir(file_path):
        file_path_date = file_path + date_name + '\\'
        data_one_day = pd.DataFrame()
        for i in range(len(os.listdir(file_path_date))):
            file_name = os.listdir(file_path_date)[i]
            file_location = file_path_date + file_name
            data = pd.read_csv(file_location, index_col=0)
            if i == 0:
                data_one_day = data.copy()
            else:
                data_one_day = pd.merge(left=data_one_day, right=data, how='outer', on=['Fut_code', 'Trade_DT'])
            print(f"{date_name}   {file_name} is ok")
        data_all = pd.concat([data_all, data_one_day])
    data_all.to_csv("1111.csv")

import os.path
import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

warnings.filterwarnings('ignore')


def tick_data_stats(tickers, date):
    data = ExtractDataPostgre.get_ts_data(tickers=tickers, start_date=date, end_date=date,
                                          key_word=['volume', 'amount', 'position', 'high', 'low', 'last'],
                                          table_name='future.t_marketdata_tick',
                                          db_engine='securitymaster', code_str='windcode', trade_dt_str='datetime',
                                          code_str_filter='windcode', trade_dt_filter_str='tradingday',
                                          other_condition=None,
                                          as_matrix=False)
    na_count = data['volume'].isna().sum()
    tick_len = len(data)
    data['datetime'] = pd.to_datetime(data['datetime'])

    min_data = ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                            key_word='close').dropna()
    min_data_length = len(min_data)
    min_data['datetime'] = pd.to_datetime(min_data['datetime'])
    first_min = min_data['datetime'].iloc[0] + datetime.timedelta(minutes=-1)
    last_min = min_data['datetime'].iloc[-1]

    non_continuous_trading_time_count = len(data[data['datetime'] < first_min]) + len(data[data['datetime'] > last_min])
    other_trading_time_count = len(data[data['datetime'] < first_min + datetime.timedelta(minutes=-30)]) + len(
        data[data['datetime'] > last_min + datetime.timedelta(minutes=30)])
    df = pd.DataFrame(
        columns=['tickers', 'date', 'tick_len', 'min_len', 'na_count', 'non_continuous_trading_time_count',
                 'other_trading_time_count'])
    df.loc[0, "tickers"] = tickers
    df.loc[0, "date"] = date
    df.loc[0, "tick_len"] = tick_len
    df.loc[0, "min_len"] = min_data_length
    df.loc[0, "na_count"] = na_count
    df.loc[0, "non_continuous_trading_time_count"] = non_continuous_trading_time_count
    df.loc[0, "other_trading_time_count"] = other_trading_time_count
    return df


def generate_max_oi_main_contract(tickers, date):
    sql_string = f"select current_main_instrumentid from future.t_oi_main_contract_map_daily where main_contract_code=\'{tickers}\' and tradingday=\'{date}\' order by tradingday DESC limit 1"
    main_contract_windcode = ExtractDataPostgre.sql_query_from_qa(sql_statement=sql_string)
    main_contract_windcode = main_contract_windcode.iloc[0, 0]
    return main_contract_windcode


def all_tick_tickers_in_one_date(date):
    sql_string = f"select windcode from future.t_marketdata_tick where tradingday=\'{date}\' group by windcode"
    windcode = ExtractDataPostgre.sql_query_from_sm(sql_statement=sql_string)
    windcode = list(windcode.iloc[:, 0])
    return windcode


if __name__ == '__main__':
    save_path = "C:\\Users\\jason.huang\\research\\data\\volume_bar_and_dollar_bar_statistics\\"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tickers_list = ConstFut.fut_code_list
    trading_calendar = ExtractDataPostgre.get_trading_calendar()
    trading_calendar.iloc[:, 0] = [datetime.datetime.strftime(x, "%Y-%m-%d") for x in trading_calendar.iloc[:, 0]]
    trading_calendar = np.array(trading_calendar.iloc[:, 0])
    trading_calendar = trading_calendar[trading_calendar >= '2022-01-01']
    trading_calendar = trading_calendar[trading_calendar < '2022-09-01']
    trading_calendar = trading_calendar[::-1]
    # tickers_list = ['TF.CFE', 'IH.CFE', 'RB.SHF', 'CU.SHF', 'TA.CZC', 'M.DCE', 'SR.CZC', 'CJ.CZC', 'FU.SHF', 'PP.DCE',
    #                 'JD.DCE', 'V.DCE']
    # date_list = ['2022-12-01', '2021-12-01', '2022-05-25']
    data_all = pd.DataFrame()
    for date in trading_calendar:
        tickers_list = all_tick_tickers_in_one_date(date)
        for tickers in tickers_list:
            try:
                # main_contract_code = generate_max_oi_main_contract(tickers, date)
                df = tick_data_stats(tickers=tickers, date=date)
                data_all = pd.concat([data_all, df])
                print(f"{date} {tickers} is ok")
            except Exception as e:
                print(f"{date} {tickers}  Reason: {e}")

            data_all.to_csv(save_path + "tick_stats.csv")


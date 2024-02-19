import os.path
import time
import warnings

import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

warnings.filterwarnings('ignore')

# tickers_list = ConstFut.fut_code_list
start_date = '2018-01-01'
end_date = '2021-09-08'

coef_of_std = 1
time_lag = 5

save_path = f"C:\\Users\\jason.huang\\research\\single_direction_market_determination\\intraday_large_volume_diff_vol\\"
daily_save_path = f"{save_path}daily_main_contract_vol_coef_of_std_{coef_of_std}_time_lag_{time_lag}\\"
if not os.path.exists(daily_save_path):
    os.mkdir(daily_save_path)


def generate_max_oi_main_contract(tickers, date):
    sql_string = f"select current_main_instrumentid from future.t_oi_main_contract_map_daily where main_contract_code=\'{tickers}\' and tradingday=\'{date}\'"
    main_contract_windcode = ExtractDataPostgre.sql_query_from_qa(sql_statement=sql_string)
    main_contract_windcode = main_contract_windcode.iloc[0, 0]
    return main_contract_windcode


if __name__ == '__main__':

    trading_calendar = np.array(ExtractDataPostgre.get_trading_calendar().iloc[:, 0])
    trading_calendar = np.array([datetime.datetime.strftime(x, "%Y-%m-%d") for x in trading_calendar])
    trading_calendar = trading_calendar[trading_calendar >= start_date]
    trading_calendar = list(trading_calendar[trading_calendar <= end_date])
    trading_calendar = trading_calendar[::-1]

    data_all = pd.DataFrame()
    for date in trading_calendar:
        data_one_date = pd.DataFrame()
        index = 0
        # tickers_list = [generate_max_oi_main_contract(x, date) for x in ConstFut.fut_code_list]
        for symbol in ConstFut.fut_code_list:

            try:
                tickers = generate_max_oi_main_contract(symbol, date)
                data = ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                                    key_word=['close', 'volume'])
                data['ret'] = data['close'].div(data['close'].shift(1)).sub(1)
                # data['volume_diff'] = data['volume'].diff()
                data['datetime'] = pd.to_datetime(data['datetime'])
                data['time'] = [x.time() for x in data['datetime']]
                trading_min = ConstFut.fut_code_trading_min_time_dict[
                    ExtractDataPostgre.get_code_instrument_mapping()[tickers]]
                trading_min_datetime = [datetime.time(int(x[0][:2]), int(x[0][-2:]) + 1) for x in trading_min] + [
                    datetime.time(int(x[1][:2]), int(x[1][-2:])) for x in trading_min]
                data = data[~data['time'].isin(trading_min_datetime)]
                data.index = range(len(data))
                data['volume_diff'] = data['volume'].diff()
                volume_diff_mean = data['volume_diff'].mean()
                volume_diff_std = data['volume_diff'].std()
                data_large_volume = data[data['volume_diff'] > volume_diff_mean + coef_of_std * volume_diff_std]
                for i in range(len(data_large_volume)):
                    data_large_volume.loc[data_large_volume.index[i], 'vol'] = data.loc[data_large_volume.index[i]:min(
                        data_large_volume.index[i] + time_lag - 1, len(data)), 'ret'].std()
                data_one_date.loc[index, 'tickers'] = tickers
                data_one_date.loc[index, 'tradingday'] = date
                data_one_date.loc[index, 'intraday_large_volume_vol'] = data_large_volume['vol'].mean()
                data_one_date.loc[index, 'intraday_large_volume_ret'] = data_large_volume['ret'].mean()
                print(f"{tickers} {date} is ok")
            except Exception as e:
                data_one_date.loc[index, 'tickers'] = symbol
                data_one_date.loc[index, 'tradingday'] = date
                data_one_date.loc[index, 'intraday_large_volume_vol'] = np.nan
                data_one_date.loc[index, 'intraday_large_volume_ret'] = np.nan
                print(f"{symbol} {date}  is not ok Reason {e}")
            index = index + 1
            # print(f"{tickers} {date} is ok")
        data_one_date.to_csv(f"{daily_save_path}{date}.csv")
        # data_all = pd.concat([data_all, data_one_date])
        # data_all.to_csv(f"{save_path}daily_main_contract_vol_coef_of_std_{coef_of_std}_time_lag_{time_lag}.csv")

    # for date in trading_calendar:
    #     for x in ConstFut.fut_code_list:
    #         try:
    #             generate_max_oi_main_contract(x, date)
    #         except Exception as e:
    #             print(f"{x} {date} {e}")
    #     print(f"{date} is ok")
    #
    #     tickers_list = [generate_max_oi_main_contract(x, date) for x in ConstFut.fut_code_list]
    #     print(f"{date}: {len(tickers_list)}")

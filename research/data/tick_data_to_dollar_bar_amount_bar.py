import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

warnings.filterwarnings('ignore')


def tick_data_to_volume_bar(tickers, date, freq_str='1min'):
    data = ExtractDataPostgre.get_ts_data(tickers=tickers, start_date=date, end_date=date,
                                          key_word=['volume', 'amount', 'position', 'high', 'low', 'last'],
                                          table_name='future.t_marketdata_tick',
                                          db_engine='securitymaster', code_str='windcode', trade_dt_str='datetime',
                                          code_str_filter='windcode', trade_dt_filter_str='tradingday',
                                          other_condition=None,
                                          as_matrix=False)
    data['volume'] = data['volume'].fillna(method='ffill').fillna(0)
    data['amount'] = data['amount'].fillna(method='ffill').fillna(0)
    data['position'] = data['position'].fillna(method='ffill')
    data['volume_diff'] = data['volume'].fillna(method='ffill').fillna(0).diff()
    data.loc[0, 'volume_diff'] = data.loc[0, 'volume']
    data['volume_diff'] = data['volume_diff'].fillna(0)

    data['amount_diff'] = data['amount'].fillna(method='ffill').fillna(0).diff()
    data.loc[0, 'amount_diff'] = data.loc[0, 'amount']
    data['amount_diff'] = data['amount_diff'].fillna(0)

    data['position_diff'] = data['position'].fillna(method='ffill').fillna(0).diff()
    data.loc[0, 'position_diff'] = data.loc[0, 'position']
    data['position_diff'] = data['position_diff'].fillna(0)
    if freq_str == '1min':
        num = len(ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                               key_word='close').dropna())
    else:
        num = len(ExtractDataPostgre.get_ts_data(tickers=tickers, start_date=date, end_date=date,
                                                 key_word=['close'],
                                                 table_name='future.t_marketdata_minute_aggregations',
                                                 db_engine='securitymaster', code_str='windcode',
                                                 trade_dt_str='datetime',
                                                 code_str_filter='windcode', trade_dt_filter_str='tradingday',
                                                 other_condition=f"agg_frequency={freq_str}",
                                                 as_matrix=False))
    result = pd.DataFrame(index=range(num),
                          columns=['windcode', 'tradingday', 'datetime', 'open', 'high', 'low', 'close', 'volume',
                                   'amount', 'position'])
    total_volume = \
        ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                    key_word=['vol']).iloc[0, -1]
    if total_volume == data['volume'].iloc[-1]:
        pass
    else:
        print("tick data volume is not equal to daily data volume")
        total_volume = data['volume'].iloc[-1]
    index_record = [0 for _ in range(num + 1)]
    for i in range(num):
        if i == num - 1:
            try:
                volume_threshold = total_volume / num * (i + 1)
                index_record[i + 1] = np.where(data['volume'] >= volume_threshold)[0][0]
            except:
                index_record[i + 1] = len(data) - 1
        else:
            volume_threshold = total_volume / num * (i + 1)
            index_record[i + 1] = np.where(data['volume'] >= volume_threshold)[0][0]
    result['windcode'] = tickers
    result['tradingday'] = date

    for i in range(num):
        result.loc[i, 'datetime'] = data['datetime'][index_record[i + 1]]
        if index_record[i + 1] == len(result) - 1:
            result.loc[i, 'open'] = data['last'][index_record[i]:].iloc[0]
            result.loc[i, 'close'] = data['last'][index_record[i]:].iloc[-1]
            result.loc[i, 'high'] = data['last'][index_record[i]:].max()
            result.loc[i, 'low'] = data['last'][index_record[i]:].min()
        else:
            result.loc[i, 'open'] = data['last'][index_record[i]:index_record[i + 1] + 1].iloc[0]
            result.loc[i, 'close'] = data['last'][index_record[i]:index_record[i + 1] + 1].iloc[-1]
            result.loc[i, 'high'] = data['last'][index_record[i]:index_record[i + 1] + 1].max()
            result.loc[i, 'low'] = data['last'][index_record[i]:index_record[i + 1] + 1].min()
        result.loc[i, 'volume'] = total_volume / num

        if index_record[i] == index_record[i + 1]:
            try:
                pct1 = (-data['volume'][index_record[i] - 1] + total_volume / num * i) / data['volume_diff'][
                    index_record[i]]
            except KeyError:
                pct1 = 1 - (data['volume'][index_record[i]] - total_volume / num * i) / data['volume_diff'][
                    index_record[i]]
            try:
                pct2 = (-data['volume'][index_record[i] - 1] + total_volume / num * (i + 1)) / data['volume_diff'][
                    index_record[i + 1]]
            except KeyError:
                pct2 = 1-(data['volume'][index_record[i]] - total_volume / num * (i + 1)) / data['volume_diff'][
                    index_record[i + 1]]

            result.loc[i, 'amount'] = data['amount_diff'][index_record[i + 1]] * (pct2 - pct1)
            result.loc[i, 'position'] = data['position'][index_record[i + 1]]

        else:
            if data['volume_diff'][index_record[i]] == 0:
                pct1 = 1
            else:
                pct1 = (data['volume'][index_record[i]] - total_volume / num * i) / data['volume_diff'][
                    index_record[i]]
            if data['volume_diff'][index_record[i + 1]] == 0:
                pct2 = 1
            else:
                pct2 = 1 - (data['volume'][index_record[i + 1]] - total_volume / num * (i + 1)) / \
                       data['volume_diff'][
                           index_record[i + 1]]

            if index_record[i + 1] - index_record[i] == 1:
                result.loc[i, 'amount'] = pct1 * data['amount_diff'][index_record[i]] + pct2 * data['amount_diff'][
                    index_record[i + 1]]
            else:
                result.loc[i, 'amount'] = pct1 * data['amount_diff'][index_record[i]] + pct2 * data['amount_diff'][
                    index_record[i + 1]] + data['amount_diff'][index_record[i] + 1:index_record[i + 1]].sum()

            result.loc[i, 'position'] =data['position_diff'][index_record[i + 1]] * pct2 + data['position'][index_record[i + 1] - 1]

    return result


def tick_data_to_dollar_bar(tickers, date, freq_str='1min'):
    data = ExtractDataPostgre.get_ts_data(tickers=tickers, start_date=date, end_date=date,
                                          key_word=['volume', 'amount', 'position', 'high', 'low', 'last'],
                                          table_name='future.t_marketdata_tick',
                                          db_engine='securitymaster', code_str='windcode', trade_dt_str='datetime',
                                          code_str_filter='windcode', trade_dt_filter_str='tradingday',
                                          other_condition=None,
                                          as_matrix=False)
    data['volume'] = data['volume'].fillna(method='ffill').fillna(0)
    data['amount'] = data['amount'].fillna(method='ffill').fillna(0)
    data['position'] = data['position'].fillna(method='ffill')
    data['volume_diff'] = data['volume'].fillna(method='ffill').fillna(0).diff()
    data.loc[0, 'volume_diff'] = data.loc[0, 'volume']
    data['volume_diff'] = data['volume_diff'].fillna(0)

    data['amount_diff'] = data['amount'].fillna(method='ffill').fillna(0).diff()
    data.loc[0, 'amount_diff'] = data.loc[0, 'amount']
    data['amount_diff'] = data['amount_diff'].fillna(0)

    data['position_diff'] = data['position'].fillna(method='ffill').fillna(0).diff()
    data.loc[0, 'position_diff'] = data.loc[0, 'position']
    data['position_diff'] = data['position_diff'].fillna(0)
    if freq_str == '1min':
        num = len(ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                               key_word='close').dropna())
    else:
        num = len(ExtractDataPostgre.get_ts_data(tickers=tickers, start_date=date, end_date=date,
                                                 key_word=['close'],
                                                 table_name='future.t_marketdata_minute_aggregations',
                                                 db_engine='securitymaster', code_str='windcode',
                                                 trade_dt_str='datetime',
                                                 code_str_filter='windcode', trade_dt_filter_str='tradingday',
                                                 other_condition=f"agg_frequency={freq_str}",
                                                 as_matrix=False))
    result = pd.DataFrame(index=range(num),
                          columns=['windcode', 'tradingday', 'datetime', 'open', 'high', 'low', 'close', 'volume',
                                   'amount', 'position'])
    total_amount = \
        ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                    key_word=['amount']).iloc[0, -1]
    total_amount = total_amount * 10000
    if int(total_amount / 10000) == int(data['amount'].iloc[-1] / 10000):
        pass
    else:
        total_amount = data['amount'].iloc[-1]
        print("Tick Amount is different with daily amount")
    index_record = [0 for _ in range(num + 1)]
    for i in range(num):
        if i == num - 1:
            amount_threshold = total_amount / num * (i + 1)
            try:
                index_record[i + 1] = np.where(data['amount'] >= amount_threshold)[0][0]
            except:
                index_record[i + 1] = len(data) - 1
        else:
            amount_threshold = total_amount / num * (i + 1)
            index_record[i + 1] = np.where(data['amount'] >= amount_threshold)[0][0]
    result['windcode'] = tickers
    result['tradingday'] = date

    # amount_threshold=total_amount / num
    for i in range(num):
        result.loc[i, 'datetime'] = data['datetime'][index_record[i + 1]]
        if index_record[i + 1] == len(result) - 1:
            result.loc[i, 'open'] = data['last'][index_record[i]:].iloc[0]
            result.loc[i, 'close'] = data['last'][index_record[i]:].iloc[-1]
            result.loc[i, 'high'] = data['last'][index_record[i]:].max()
            result.loc[i, 'low'] = data['last'][index_record[i]:].min()
        else:
            result.loc[i, 'open'] = data['last'][index_record[i]:index_record[i + 1] + 1].iloc[0]
            result.loc[i, 'close'] = data['last'][index_record[i]:index_record[i + 1] + 1].iloc[-1]
            result.loc[i, 'high'] = data['last'][index_record[i]:index_record[i + 1] + 1].max()
            result.loc[i, 'low'] = data['last'][index_record[i]:index_record[i + 1] + 1].min()
        result.loc[i, 'amount'] = total_amount / num

        if index_record[i] == index_record[i + 1]:

            try:
                pct1 = (-data['amount'][index_record[i] - 1] + total_amount / num * i) / data['amount_diff'][
                    index_record[i]]
            except KeyError:
                pct1 = 1 - (data['amount'][index_record[i]] - total_amount / num * i) / data['amount_diff'][
                    index_record[i]]
            try:
                pct2 = (-data['amount'][index_record[i] - 1] + total_amount / num * (i + 1)) / data['amount_diff'][
                    index_record[i + 1]]
            except KeyError:
                pct2 = 1-(data['amount'][index_record[i]] - total_amount / num * (i + 1)) / data['amount_diff'][
                    index_record[i + 1]]

            result.loc[i, 'volume'] = data['volume_diff'][index_record[i + 1]] * (pct2 - pct1)
            result.loc[i, 'position'] = data['position'][index_record[i + 1]]

        else:
            if data['amount_diff'][index_record[i]] == 0:
                pct1 = 1
            else:
                pct1 = (data['amount'][index_record[i]] - total_amount / num * i) / data['amount_diff'][
                    index_record[i]]
            if data['amount_diff'][index_record[i + 1]] == 0:
                pct2 = 1
            else:
                pct2 = 1 - (data['amount'][index_record[i + 1]] - total_amount / num * (i + 1)) / \
                       data['amount_diff'][
                           index_record[i + 1]]

            if index_record[i + 1] - index_record[i] == 1:
                result.loc[i, 'volume'] = pct1 * data['volume_diff'][index_record[i]] + pct2 * data['volume_diff'][
                    index_record[i + 1]]
            else:
                result.loc[i, 'volume'] = pct1 * data['volume_diff'][index_record[i]] + pct2 * data['volume_diff'][
                    index_record[i + 1]] + data['volume_diff'][index_record[i] + 1:index_record[i + 1]].sum()

            result.loc[i, 'position'] = data['position_diff'][index_record[i + 1]] * pct2 + data['position'][index_record[i + 1] - 1]

    return result


if __name__ == '__main__':
    save_path = "C:\\Users\\jason.huang\\research\\data\\volume_bar_and_dollar_bar\\"

    tickers_list = ['TF.CFE', 'IH.CFE', 'RB.SHF', 'CU.SHF', 'TA.CZC', 'M.DCE', 'SR.CZC', 'CJ.CZC', 'FU.SHF', 'PP.DCE',
                    'JD.DCE', 'V.DCE']
    date_list = ['2022-12-01', '2021-12-01', '2022-05-25']

    for tickers_code in tickers_list:
        for date in date_list:
            contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers_code, start_date=date,
                                                                end_date=date, index=1,
                                                                roll_method='oi')
            tickers = contract_df.iloc[0, -1]
            try:
                volume_bar = tick_data_to_volume_bar(tickers=tickers, date=date)
                volume_bar.to_csv(f"{save_path}{tickers}{date}_volume_bar.csv")
                print(f"{tickers} {date}_volume_bar is ok")
            except Exception as e:

                print(f"{tickers} {date}_volume_bar is not ok reason is {e}")

            try:
                dollar_bar = tick_data_to_dollar_bar(tickers=tickers, date=date)
                dollar_bar.to_csv(f"{save_path}{tickers}{date}_dollar_bar.csv")
                print(f"{tickers} {date}_dollar_bar is ok")
            except Exception as e:
                print(f"{tickers} {date}_dollar_bar is not ok  reason is {e}")

import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

warnings.filterwarnings('ignore')


def min_data_aggregation(tickers, date, freq='5min'):
    freq_num = int(freq[:-3])
    minute_data = ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers, start_date=date, end_date=date,
                                                               key_word=['open', 'high', 'low', 'close', 'volume',
                                                                         'amount', 'position'])
    minute_data['group'] = [i // freq_num for i in range(len(minute_data))]
    minute_data['vwap'] = minute_data['amount'].div(
        ConstFut.fut_code_bpv[tickers.split('.')[0][:-4] + '.' + tickers.split('.')[1]]) / minute_data['volume']
    open_data = minute_data.groupby('group')['open'].first()
    close_data = minute_data.groupby('group')['close'].last()
    high_data = minute_data.groupby('group')['high'].max()
    low_data = minute_data.groupby('group')['low'].min()
    volume_data = minute_data.groupby('group')['volume'].sum()
    amount_data = minute_data.groupby('group')['amount'].sum()
    position_data = minute_data.groupby('group')['position'].last()
    position_range_data = (minute_data.groupby('group')['position'].last() - minute_data.groupby('group')[
        'position'].first()).abs()
    twap_data = minute_data.groupby('group')['close'].mean()
    t_vwap_data = minute_data.groupby('group')['vwap'].mean()
    timestamp_list = minute_data.groupby('group')['datetime'].last()
    data_all = pd.concat(
        [timestamp_list, open_data, high_data, low_data, close_data, volume_data, amount_data, position_data, twap_data,
         position_range_data, t_vwap_data], axis=1)
    data_all.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'position', 'twap',
                        'position_range', 't_vwap']
    data_all['windcode'] = tickers
    data_all['agg_frequency'] = freq
    data_all = data_all[['windcode', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'position', 'twap',
                         'position_range', 't_vwap', 'agg_frequency']]
    return data_all


if __name__ == '__main__':
    save_path = "C:\\Users\\jason.huang\\research\\data\\min_aggregation\\"

    tickers_list = ['TF.CFE', 'IH.CFE', 'RB.SHF', 'CU.SHF', 'TA.CZC', 'M.DCE', 'SR.CZC', 'CJ.CZC', 'FU.SHF', 'PP.DCE',
                    'JD.DCE', 'V.DCE']
    date_list = ['2022-12-01', '2021-12-01', '2022-05-25']
    freq = '5min'

    for tickers_code in tickers_list:
        for date in date_list:
            contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers_code, start_date=date,
                                                                end_date=date, index=1,
                                                                roll_method='oi')
            tickers = contract_df.iloc[0, -1]
            try:
                data = min_data_aggregation(tickers, date, freq=freq)
                data.to_csv(f"{save_path}{tickers}_agg_frequenct_{freq}.csv")
                print(f"{tickers} {date} {freq} is  ok")
            except Exception as e:
                print(f"{tickers} {date} {freq} is  ok   Reason is {e}")

import os
import warnings

import pandas as pd
import numpy as np
import datetime
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import model.constants.futures as ConstFutBasic
import model.constants.path as ConstPath

import time

warnings.filterwarnings('ignore')


def update_crypto_future_hourly_feature_data(tickers_list=None, start_date=None, end_date=None, table_name=None,
                                             key_word_list=None, feature_save_path=None):
    if table_name is None:
        table_name = 'crypto.t_marketdata_future_hourly_binance'
    if tickers_list is None:
        tickers_list = ExtractDataPostgre.get_crypto_tickers_list(exchange='binance', table_name=table_name)
        tickers_list = [x for x in tickers_list if 'USDT' in x]
    if start_date is None:
        start_date = '2023-01-01'
    if end_date is None:
        end_date = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")
    if key_word_list is None:
        key_word_list = ['open', 'high', 'low', 'close', 'volume', 'base_volume', 'trade_count']
    if feature_save_path is None:
        feature_save_path = "C:\\Users\\jason.huang\\research\\scripts_working\\GP_tracking\\crypto\\crypto_binance_future_hourly\\Input_Feature\\"

    if not os.path.exists(feature_save_path):
        try:
            os.makedirs(feature_save_path)
        except:
            try:
                os.mkdir(feature_save_path)
            except Exception as e:
                print(e)

    for key_word in key_word_list:

        data = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date,
                                                              end_date=end_date, key_word=key_word, as_matrix=True,
                                                              table_name=table_name)
        data = data.fillna(method='ffill', limit=10)
        data.index.name = 'Trade_DT'
        if key_word=='base_volume':
            data.to_csv(feature_save_path + 'freq_1H_' + 'amount' + '.csv')
        else:
            data.to_csv(feature_save_path + 'freq_1H_' + key_word + '.csv')
        print(key_word + ' is ok')

    close_price = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date,
                                                             end_date=end_date, key_word='close', as_matrix=True,
                                                             table_name=table_name)
    close_price = close_price.fillna(method='ffill', limit=10)
    pct1 = close_price.div(close_price.shift(1)) - 1
    pct1.index.name = 'Trade_DT'
    pct1.to_csv(feature_save_path + 'freq_1H_pct1.csv')
    print('pct1 is ok')

    pct1.to_csv(feature_save_path + 'freq_1H_close_ret.csv')
    print('close_ret is ok')

    open_price = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date,
                                                            end_date=end_date,
                                                            key_word='open', as_matrix=True, table_name=table_name)
    open_price = open_price.fillna(method='ffill', limit=10)
    pct2 = open_price.div(open_price.shift(1)) - 1
    pct2.index.name = 'Trade_DT'
    pct2.to_csv(feature_save_path + 'freq_1H_pct2.csv')
    print('pct2 is ok')

    high_price = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date,
                                                            end_date=end_date, key_word='high', as_matrix=True,
                                                            table_name=table_name).fillna(method='ffill', limit=10)
    low_price = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date,
                                                           end_date=end_date, key_word='low', as_matrix=True,
                                                           table_name=table_name).fillna(method='ffill', limit=10)

    pct3 = high_price.div(low_price) - 1
    pct3.index.name = 'Trade_DT'
    pct3.to_csv(feature_save_path + 'freq_1H_pct3.csv')
    print('pct3 is ok')

    pct4 = close_price.div(open_price) - 1
    pct4.index.name = 'Trade_DT'
    pct4.to_csv(feature_save_path + 'freq_1H_pct4.csv')
    print('pct4 is ok')

    # check shape
    for file_name in os.listdir(feature_save_path):
        file_location = feature_save_path + file_name
        data = pd.read_csv(file_location)
        print(f"{file_name} : {data.shape}")

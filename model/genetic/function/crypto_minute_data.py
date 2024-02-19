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

tickers_list = ExtractDataPostgre.get_crypto_tickers_list(exchange='binance',
                                                          table_name='crypto.t_marketdata_minute')
tickers_list = [x for x in tickers_list if 'USDT' in x]
start_date = '2022-10-01'
end_date = '2022-11-30'
key_word_list = ['open', 'high', 'low', 'close', 'volume', 'base_volume', 'trade_count']
feature_save_path = 'C:\\Users\\jason.huang\\research\\data_mining\\GP\\features\\crypto_feature_minute_2month\\'
if not os.path.exists(feature_save_path):
    os.makedirs(feature_save_path)
# ret
close = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='close', as_matrix=True,
                                                   table_name='crypto.t_marketdata_minute')
close = close.fillna(method='ffill', limit=10)
pct1 = close.div(close.shift(1)) - 1
pct1.index.name = 'Trade_DT'
pct1.to_csv(feature_save_path + 'freq_1min_close_ret.csv')
print('close ret is ok')

for key_word in key_word_list:
    data = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                      key_word=key_word, as_matrix=True,
                                                      table_name='crypto.t_marketdata_minute')
    data = data.fillna(method='ffill', limit=10)
    data.index.name = 'Trade_DT'
    data.to_csv(feature_save_path + 'freq_1min_' + key_word + '.csv')
    print(key_word + ' is ok')

close = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='close', as_matrix=True,
                                                   table_name='crypto.t_marketdata_minute')
close = close.fillna(method='ffill', limit=10)
pct1 = close.div(close.shift(1)) - 1
pct1.index.name = 'Trade_DT'
pct1.to_csv(feature_save_path + 'freq_1min_pct1.csv')
print('pct1 is ok')

open = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                  key_word='open', as_matrix=True,
                                                  table_name='crypto.t_marketdata_minute')
open = open.fillna(method='ffill', limit=10)
pct2 = open.div(open.shift(1)) - 1
pct2.index.name = 'Trade_DT'
pct2.to_csv(feature_save_path + 'freq_1min_pct2.csv')
print('pct2 is ok')

high = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                  key_word='high', as_matrix=True,
                                                  table_name='crypto.t_marketdata_minute').fillna(method='ffill',
                                                                                                         limit=10)
low = ExtractDataPostgre.get_crypto_hour_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                 key_word='low', as_matrix=True,
                                                 table_name='crypto.t_marketdata_minute').fillna(method='ffill',
                                                                                                        limit=10)

pct3 = high.div(low) - 1
pct3.index.name = 'Trade_DT'
pct3.to_csv(feature_save_path + 'freq_1min_pct3.csv')
print('pct3 is ok')

pct4 = close.div(open) - 1
pct4.index.name = 'Trade_DT'
pct4.to_csv(feature_save_path + 'freq_1min_pct4.csv')
print('pct4 is ok')

# check shape
for file_name in os.listdir(feature_save_path):
    file_location = feature_save_path + file_name
    data = pd.read_csv(file_location)
    print(f"{file_name} : {data.shape}")



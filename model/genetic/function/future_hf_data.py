import os
import warnings

import pandas as pd

import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import model.constants.futures as ConstFutBasic
import model.constants.path as ConstPath

warnings.filterwarnings('ignore')

# tickers_list = ConstFutBasic.fut_code_list
tickers_list = ['IC.CFE', 'IF.CFE', 'IH.CFE']
start_date = '2018-01-01'
end_date = '2022-09-30'
freq = '30min'
save_path = 'C:\\Users\\jason.huang\\research\data\\freq_30min\\'
# predict percent
pct = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                  key_word='close', freq=freq, index=1, ret_index=True)
pct['Fut_code'] = pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct = pct.pivot_table(index='Trade_DT', columns='Fut_code', values=pct.columns[2])
pct.to_csv(f"{save_path}freq_{freq}_close_ret.csv")

# basic price volume factor

close = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                    key_word='close', freq=freq, index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close.to_csv(f"{save_path}freq_{freq}_close.csv")

open = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='open', freq=freq, index=1, ret_index=False)
open['Fut_code'] = open['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open = open.pivot_table(index='Trade_DT', columns='Fut_code', values=open.columns[2])
open.to_csv(f"{save_path}freq_{freq}_open.csv")

high = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='high', freq=freq, index=1, ret_index=False)
high['Fut_code'] = high['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
high = high.pivot_table(index='Trade_DT', columns='Fut_code', values=high.columns[2])
high.to_csv(f"{save_path}freq_{freq}_high.csv")

low = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                  key_word='low',
                                                  freq=freq, index=1, ret_index=False)
low['Fut_code'] = low['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
low = low.pivot_table(index='Trade_DT', columns='Fut_code', values=low.columns[2])
low.to_csv(f"{save_path}freq_{freq}_low.csv")

volume = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='volume', freq=freq, index=1, ret_index=False)
volume['Fut_code'] = volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume = volume.pivot_table(index='Trade_DT', columns='Fut_code', values=volume.columns[2])
volume.to_csv(f"{save_path}freq_{freq}_volume.csv")

amount = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='amount', freq=freq, index=1, ret_index=False)
amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])
amount.to_csv(f"{save_path}freq_{freq}_amount.csv")

position = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                       key_word='position', freq=freq, index=1, ret_index=False)
position['Fut_code'] = position['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
position = position.pivot_table(index='Trade_DT', columns='Fut_code', values=position.columns[2])
position.to_csv(f"{save_path}freq_{freq}_oi.csv")

twap = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='twap', freq=freq, index=1, ret_index=False)
twap['Fut_code'] = twap['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
twap = twap.pivot_table(index='Trade_DT', columns='Fut_code', values=twap.columns[2])
twap.to_csv(f"{save_path}freq_{freq}_twap.csv")

position_range = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                             end_date=end_date, key_word='position_range', freq=freq,
                                                             index=1, ret_index=False)
position_range['Fut_code'] = position_range['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
position_range = position_range.pivot_table(index='Trade_DT', columns='Fut_code', values=position_range.columns[2])
position_range.to_csv(f"{save_path}freq_{freq}_position_range.csv")

t_vwap = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='t_vwap', freq=freq, index=1, ret_index=False)
t_vwap['Fut_code'] = t_vwap['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
t_vwap = t_vwap.pivot_table(index='Trade_DT', columns='Fut_code', values=t_vwap.columns[2])
t_vwap.to_csv(f"{save_path}freq_{freq}_t_vwap.csv")

# pct 1 close T / close T-1  -1
# pct 2 open T / open T-1    -1
# pct 3  high T / low T      -1
# pct 4  close T / open T    -1

pct1 = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='close', freq='halfDay', index=1, ret_index=True)
pct1['Fut_code'] = pct1['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct1 = pct1.pivot_table(index='Trade_DT', columns='Fut_code', values=pct1.columns[2])

pct1.to_csv(f"{save_path}freq_{freq}_pct1.csv")

pct2 = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='open', freq='halfDay', index=1, ret_index=True)
pct2['Fut_code'] = pct2['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct2 = pct2.pivot_table(index='Trade_DT', columns='Fut_code', values=pct2.columns[2])
pct2.to_csv(f"{save_path}freq_{freq}_pct2.csv")
pct3 = high.div(low) - 1
pct3.to_csv(f"{save_path}freq_{freq}_pct3.csv")
pct4 = close.div(open) - 1
pct4.to_csv(f"{save_path}freq_{freq}_pct4.csv")

for file in os.listdir(save_path):
    data = pd.read_csv(save_path + file)
    print(f"{file}: {data.shape}")

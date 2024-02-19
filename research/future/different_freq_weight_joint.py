import pandas as pd
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

file_path = 'C:\\Users\\jason.huang\\research\\GP\\crypto_minute\\data\\'

weight_1H_file_location = file_path + 'crypto_spot_binance_1H_train202109_202209.csv'
weight_5min_file_location = file_path + 'crypto_spot_binance_5min_train202210_202211.csv'

close_1H=pd.read_csv(file_path +'freq_1H_close.csv',index_col=0)
close_5min=pd.read_csv(file_path +'freq_5min_close.csv',index_col=0)


weight_1h = pd.read_csv(weight_1H_file_location, index_col=0)
weight_1h.index = pd.to_datetime(weight_1h.index)
weight_1h.columns=close_1H.columns
weight_5min = pd.read_csv(weight_5min_file_location, index_col=0)
weight_5min.index = pd.to_datetime(weight_5min.index)
weight_5min.columns=close_5min.columns
weight_1h=weight_1h[[x for x in weight_1h.columns if x in weight_5min.columns]]

trade_1h_5min = weight_5min.copy()
trade_1h_5min.iloc[:, :] = np.nan

trade_1h=weight_1h.diff().fillna(0)
trade_5min=weight_5min.diff().fillna(0)

for i in range(len(trade_1h_5min)):
    if trade_1h_5min.index[i].minute == 0:
        trade_1h_5min.iloc[i, :] = np.array(trade_1h.loc[trade_1h_5min.index[i],])
trade_1h_5min=trade_1h_5min.div(12)
trade_1h_5min=trade_1h_5min.fillna(method='bfill')

trade_5min_all=trade_5min.div(12)+trade_1h_5min
weight_5min_all=trade_5min_all.cumsum()

weight_5min_all.to_csv(file_path+'freq_5min_compositive_freq_1H_weight_df.csv')
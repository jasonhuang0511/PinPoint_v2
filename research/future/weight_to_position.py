import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import pickle
from backtest.Backtest_Object import CashBtObj, CashBt

warnings.filterwarnings("ignore")
bt_tickers_str = 'crypto_future_binance_1H_train_202109_202209'
# weight_df = pd.read_csv("C:\\Users\\jason.huang\\research\\GP_momemntum\\crypto_bin.csv", index_col=0)
# time_index = pd.to_datetime(weight_df.index)
weight_df = pd.read_csv(
    "C:\\Users\\jason.huang\\research\\GP\\crypto_future\\data\\crypto_future_binance_1H_train_202109_202209.csv",
    index_col=0)

close_df = pd.read_csv("C:\\Users\\jason.huang\\research\\GP\\crypto_future\\data\\freq_1H_close.csv",
                       index_col=0)
dollar_volume_df = pd.read_csv(
    "C:\\Users\\jason.huang\\research\\GP\\crypto_future\\data\\freq_1H_amount.csv", index_col=0)
dollar_volume_df.index = pd.to_datetime(dollar_volume_df.index)
volume_df = pd.read_csv(
    "C:\\Users\\jason.huang\\research\\GP\\crypto_future\\data\\freq_1H_volume.csv", index_col=0)
volume_df.index = pd.to_datetime(volume_df.index)
#
vwap = dollar_volume_df.div(volume_df)
#
# close_df = vwap.copy()
close_df.index = pd.to_datetime(close_df.index)
weight_df.index = pd.to_datetime(weight_df.index)

# weight_df.index = close_df.index
weight_df.columns = close_df.columns
close_df = close_df[close_df.index.isin(weight_df.index)]
dollar_volume_df = dollar_volume_df[dollar_volume_df.index.isin(weight_df.index)]
# dollar_volume_df = dollar_volume_df.mul(10000)
volume_df = volume_df[volume_df.index.isin(weight_df.index)]

# dollar_volume_df = dollar_volume_df[dollar_volume_df.index.isin(weight_df.index)]

# weight_df.columns = close_df.columns
# bpv = [200, 300, 300]
# close_df = close_df.mul(bpv)
pct_df = close_df.div(close_df.shift(1)) - 1

out_of_sample_date = '2022-09-01'
aum = 0.01 * 10000 * 10000
fee_rate_list = [0 / 100 / 100, 3/100/100,4 / 100 / 100]
index = 8757
# weight_df_long = weight_df.applymap(lambda x: x if x > 0 else 0)
weight_df_long = weight_df.copy()
# threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
mode = 'constraint_ADV'
trade_day_num = 365
intraday_freq = 24

# threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.4,0.5,0.6]

pickle_result_file_path = f"C:\\Users\\jason.huang\\research\\GP\\crypto_future\\pickle_result\\{bt_tickers_str}_aum_1m\\"
if not os.path.exists(pickle_result_file_path):
    os.mkdir(pickle_result_file_path)

for threshold in threshold_list:
    print(f"{threshold} starts")
    origin_weight_df = pd.DataFrame(weight_df_long.iloc[:, :].copy())
    slow_weight_df = pd.DataFrame(origin_weight_df.copy())

    for j in range(len(slow_weight_df.columns)):
        for i in range(len(slow_weight_df)):
            if i == 0:
                pass
            else:
                before_weight = slow_weight_df.iloc[i - 1, j]
                target_weight = origin_weight_df.iloc[i, j]
                if target_weight > before_weight:
                    if target_weight > before_weight + threshold:
                        slow_weight_df.iloc[i, j] = before_weight + threshold
                    else:
                        slow_weight_df.iloc[i, j] = target_weight
                else:
                    if target_weight < before_weight - threshold:
                        slow_weight_df.iloc[i, j] = before_weight - threshold
                    else:
                        slow_weight_df.iloc[i, j] = target_weight

    print(f"{threshold} weight is ok")

    for fee_rate in fee_rate_list:
        a = CashBt(weight_df=slow_weight_df, price_df=close_df, out_of_sample_date=out_of_sample_date,
                   fee_rate=fee_rate,
                   aum=aum, trade_price_df=close_df, mode=mode, trade_day_num=trade_day_num,
                   intraday_freq=intraday_freq,
                   amount_df=dollar_volume_df)
        a.save_result(
            f"{pickle_result_file_path}{bt_tickers_str}_all_sample_fee_{int(fee_rate * 100 * 100 * 10)}_turnover_threshold_{int(threshold * 100)}.pkl")
        print(f"{threshold} all sample is ok")

        a = CashBt(weight_df=slow_weight_df.iloc[:index, :], price_df=close_df.iloc[:index, :],
                   out_of_sample_date=out_of_sample_date, fee_rate=fee_rate,
                   aum=aum, trade_price_df=close_df.iloc[:index, :], mode=mode, trade_day_num=trade_day_num,
                   intraday_freq=intraday_freq, amount_df=dollar_volume_df.iloc[:index, :])
        a.save_result(
            f"{pickle_result_file_path}{bt_tickers_str}_fee_in_sample_{int(fee_rate * 100 * 100 * 10)}_turnover_threshold_{int(threshold * 100)}.pkl")
        print(f"{threshold} in sample is ok")

        a = CashBt(weight_df=slow_weight_df.iloc[index:, :], price_df=close_df.iloc[index:, :],
                   out_of_sample_date=out_of_sample_date, fee_rate=fee_rate,
                   aum=aum, trade_price_df=close_df.iloc[index:, :], mode=mode, trade_day_num=trade_day_num,
                   intraday_freq=intraday_freq, amount_df=dollar_volume_df.iloc[index:, :])
        a.save_result(
            f"{pickle_result_file_path}{bt_tickers_str}_fee_out_of_sample_{int(fee_rate * 100 * 100 * 10)}_turnover_threshold_{int(threshold * 100)}.pkl")
        print(f"{threshold} out of sample is ok")

# weight_df_long = weight_df.copy()
# # threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# threshold_list = [2, 3, 4, 5, 10]
#
# pickle_result_file_path = f"C:\\Users\\jason.huang\\research\\GP\\crypto_minute\\pickle_result\\{bt_tickers_str}_weight_ema\\"
# if not os.path.exists(pickle_result_file_path):
#     os.mkdir(pickle_result_file_path)
#
# for threshold in threshold_list:
#     print(f"{threshold} starts")
#     origin_weight_df = pd.DataFrame(weight_df_long.iloc[:, :].copy())
#     # slow_weight_df = pd.DataFrame(origin_weight_df.copy())
#
#     slow_weight_df = origin_weight_df.ewm(halflife=threshold).mean()
#     print(f"{threshold} weight is ok")
#
#     for fee_rate in fee_rate_list:
#         a = CashBt(weight_df=slow_weight_df, price_df=close_df, out_of_sample_date=out_of_sample_date,
#                    fee_rate=fee_rate,
#                    aum=aum, trade_price_df=close_df, mode=mode, trade_day_num=trade_day_num, intraday_freq=intraday_freq,
#                    amount_df=dollar_volume_df)
#         a.save_result(
#             f"{pickle_result_file_path}{bt_tickers_str}_all_sample_fee_{int(fee_rate * 100 * 100 * 10)}_turnover_ewm_halflife_{int(threshold)}.pkl")
#         print(f"{threshold} all sample is ok")
#
#         a = CashBt(weight_df=slow_weight_df.iloc[:index, :], price_df=close_df.iloc[:index, :],
#                    out_of_sample_date=out_of_sample_date, fee_rate=fee_rate,
#                    aum=aum, trade_price_df=close_df.iloc[:index, :], mode=mode, trade_day_num=trade_day_num,
#                    intraday_freq=intraday_freq, amount_df=dollar_volume_df.iloc[:index, :])
#         a.save_result(
#             f"{pickle_result_file_path}{bt_tickers_str}_fee_in_sample_{int(fee_rate * 100 * 100 * 10)}_turnover_ewm_halflife_{int(threshold)}.pkl")
#         print(f"{threshold} in sample is ok")
#
#         a = CashBt(weight_df=slow_weight_df.iloc[index:, :], price_df=close_df.iloc[index:, :],
#                    out_of_sample_date=out_of_sample_date, fee_rate=fee_rate,
#                    aum=aum, trade_price_df=close_df.iloc[index:, :], mode=mode, trade_day_num=trade_day_num,
#                    intraday_freq=intraday_freq, amount_df=dollar_volume_df.iloc[index:, :])
#         a.save_result(
#             f"{pickle_result_file_path}{bt_tickers_str}_fee_out_of_sample_{int(fee_rate * 100 * 100 * 10)}_turnover_ewm_halflife_{int(threshold)}.pkl")
#         print(f"{threshold} out of sample is ok")

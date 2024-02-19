import time
import warnings

import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.ConstantData.future_basic_information as ConstFut

warnings.filterwarnings('ignore')

tickers_list = ConstFut.fut_code_list
start_date = '2018-01-01'
end_date = '2023-01-01'


def close_reg_nonlinear(data):
    t = np.array([i for i in range(1, len(data) + 1)])
    t2 = np.array([i * i for i in range(1, len(data) + 1)])
    X = np.column_stack([t, t2])
    X = sm.add_constant(X)
    model = sm.OLS(data, X)
    results = model.fit()
    if 0 < results.rsquared < 1:
        return results.params[2] * results.rsquared
    else:
        return results.params[2]


def phi(x):
    return x * np.exp(-x * x / 4) / 0.89


def close_reg(data):
    t = np.array([i for i in range(1, len(data) + 1)])
    t2 = np.array([i * i for i in range(1, len(data) + 1)])
    X = np.column_stack([t, t2])
    X = sm.add_constant(X)
    model = sm.OLS(data, X)
    results = model.fit()
    if 0 < results.rsquared < 1:
        return results.params[1] * results.rsquared
    else:
        return results.params[1]


close_price_df = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date=start_date,
                                                             end_date=end_date, key_word='close', freq='Daily',
                                                             index=1, ret_index=False, roll_method='oi')
close_price_df['Fut_code'] = close_price_df['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close_price_df = close_price_df.pivot_table(index='Trade_DT', columns='Fut_code', values=close_price_df.columns[2])
close_price_df = close_price_df.fillna(method='ffill')
# close_price_df=close_price_df.iloc[:200,:2]

save_path = 'C:\\Users\\jason.huang\\research\\close_reg\\future\\nonlinear\\'
for roll_time in [5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 150, 252]:
    a = time.time()
    ts_mom = close_price_df.rolling(roll_time).apply(close_reg_nonlinear)
    ts_mom.to_csv(f"{save_path}\\close_reg_signal\\ts_mom_roll_{roll_time}.csv")
    print(time.time() - a)

    pct = close_price_df.div(close_price_df.shift(1)).sub(1)

    r1 = ts_mom.applymap(np.sign).shift(2).mul(pct).cumsum()
    r1.to_csv(f"{save_path}\\ret\\ts_mom_roll_{roll_time}_sign.csv")
    r2 = ts_mom.applymap(np.tanh).shift(2).mul(pct).cumsum()
    r2.to_csv(f"{save_path}\\ret\\ts_mom_roll_{roll_time}_tanh.csv")
    r3 = ts_mom.applymap(phi).shift(2).mul(pct).cumsum()
    r3.to_csv(f"{save_path}\\ret\\ts_mom_roll_{roll_time}_phi.csv")

    nv_all = pd.DataFrame()
    nv_all['sign_equal_weight'] = r1.diff().mean(axis=1).cumsum()
    nv_all['tanh_equal_weight'] = r2.diff().mean(axis=1).cumsum()
    nv_all['phi_equal_weight'] = r3.diff().mean(axis=1).cumsum()
    for vol_roll in [30, 60, 90, 120, 252, 512]:
        vol1 = close_price_df.div(close_price_df.shift(1)).sub(1).rolling(vol_roll).std().applymap(
            lambda x: 0.05 / (x + 0.0001) / 15.87 if x > 0.0005 else 0)
        nv_all[f"sign_vol_adjust_{vol_roll}"] = vol1.shift(1).mul(r1.diff()).sum(axis=1).cumsum()
        vol2 = close_price_df.div(close_price_df.shift(1)).sub(1).rolling(vol_roll).std().applymap(
            lambda x: 0.05 / (x + 0.0001) / 15.87 if x > 0.0005 else 0)
        nv_all[f"tanh_vol_adjust_{vol_roll}"] = vol2.shift(1).mul(r2.diff()).sum(axis=1).cumsum()
        vol3 = close_price_df.div(close_price_df.shift(1)).sub(1).rolling(vol_roll).std().applymap(
            lambda x: 0.05 / (x + 0.0001) / 15.87 if x > 0.0005 else 0)
        nv_all[f"phi_vol_adjust_{vol_roll}"] = vol3.shift(1).mul(r3.diff()).sum(axis=1).cumsum()
    nv_all.to_csv(f"{save_path}\\ret\\ts_mom_roll_{roll_time}_all.csv")

# min data
close_price_df = ExtractDataPostgre.get_continuous_future_ts(tickers=tickers_list, start_date='2018-01-01',
                                                             end_date='2023-01-01', key_word='close', freq='halfDay',
                                                             index=1, ret_index=False, roll_method='oi')
close_price_df['Fut_code'] = close_price_df['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())

for roll_time in [5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 150, 252]:
    data_all = pd.DataFrame()
    for tickers in tickers_list:
        # data=pd.DataFrame()
        close_price_one_ticker = close_price_df[close_price_df['Fut_code'] == tickers]

        # time_list=close_price_one_ticker['Trade_DT']
        close_price_one_ticker['ts_mom'] = close_price_one_ticker['close'].rolling(roll_time).apply(close_reg)
        close_price_one_ticker['sign'] = close_price_one_ticker['ts_mom'].apply(np.sign).shift(2).mul(
            close_price_one_ticker['close'].div(close_price_one_ticker['close'].shift(1)).sub(1)).cumsum()
        close_price_one_ticker['tanh'] = close_price_one_ticker['ts_mom'].apply(np.tanh).shift(2).mul(
            close_price_one_ticker['close'].div(close_price_one_ticker['close'].shift(1)).sub(1)).cumsum()
        close_price_one_ticker['phi'] = close_price_one_ticker['ts_mom'].apply(phi).shift(2).mul(
            close_price_one_ticker['close'].div(close_price_one_ticker['close'].shift(1)).sub(1)).cumsum()
        data_all = pd.concat([data_all, close_price_one_ticker])
        print(f"{tickers} is ok of roll time:{roll_time}")
    data_all.to_csv(f"{save_path}\\intraday_30min\\ts_mom_roll_{roll_time}_all_tickers.csv")

file_location = r"C:\Users\jason.huang\research\close_reg\intraday_30min\ts_mom_roll_5_all_tickers.csv"
data = pd.read_csv(file_location, index_col=0)
for tickers in tickers_list:
    data_one_ticker = data[data['Fut_code'] == tickers]
    for vol_roll in [30, 60, 90, 120, 252, 512]:
        data_one_ticker[f"vol_{vol_roll}"] = data_one_ticker['close'].rolling(vol_roll).std().apply(
            lambda x: 0.05 / (x + 0.0001) / 15.87 if x > 0.00005 else 0)
        data_one_ticker[f"sign_vol_{vol_roll}"] = data_one_ticker[f"vol_{vol_roll}"].shift(1).mul(
            data_one_ticker['ts_mom'].apply(np.sign)).shift(2).mul(
            data_one_ticker['close'].div(data_one_ticker['close'].shift(1)).sub(1)).cumsum()
        data_one_ticker[f"sign_vol_{vol_roll}"] = data_one_ticker[f"vol_{vol_roll}"].shift(1).mul(
            data_one_ticker['ts_mom'].apply(np.tanh)).shift(2).mul(
            data_one_ticker['close'].div(data_one_ticker['close'].shift(1)).sub(1)).cumsum()
        data_one_ticker[f"phi_vol_{vol_roll}"] = data_one_ticker[f"vol_{vol_roll}"].shift(1).mul(
            data_one_ticker['ts_mom'].apply(phi)).shift(2).mul(
            data_one_ticker['close'].div(data_one_ticker['close'].shift(1)).sub(1)).cumsum()

import pandas as pd
import numpy as np
import datetime

import model.factor.technical_indicator as TechIndicator
from my_stats.time_series import time_series_regression as TimeSeriesReg
import data.Process.fut_data_process as DataProcessFut
import data.ConstantData.future_basic_information as ConstFutBasic
import data.SQL.extract_data_from_postgre as ExtractDataPostgre

# warehouse_data = ExtractDataPostgre.get_ware_house(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
#                                                    end_date='2022-08-31', as_matrix=True)
# cashbasis_data = ExtractDataPostgre.get_cash_basis(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
#                                                    end_date='2022-08-31', as_matrix=True)
# cashbasis_data.index = [
#     datetime.date(int(a.strftime('%Y-%m-%d')[:4]), int(a.strftime('%Y-%m-%d')[5:7]), int(a.strftime('%Y-%m-%d')[-2:]))
#     for a in cashbasis_data.index]
# price_df = ExtractDataPostgre.get_syn_con_ts(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
#                                              end_date='2022-08-31', key_word='close')


def detrend_test(df):
    length = 300
    df_A = pd.DataFrame(index=df.index, columns=df.columns)
    df_B = pd.DataFrame(index=df.index, columns=df.columns)
    df_C = pd.DataFrame(index=df.index, columns=df.columns)
    df_theta = pd.DataFrame(index=df.index, columns=df.columns)
    df_mu = pd.DataFrame(index=df.index, columns=df.columns)
    df_sigma = pd.DataFrame(index=df.index, columns=df.columns)

    for j in range(len(df.columns)):
        for i in range(length, len(df)):
            data = df.iloc[i - length:i, j].fillna(method='ffill').dropna()
            if len(data) < 100:
                A = B = C = theta = mu = sigma = np.nan
            else:
                A, B, C, theta, mu, sigma = TimeSeriesReg.ou_process_calibration(data, n=10)
            df_A.iloc[i, j] = A
            df_B.iloc[i, j] = B
            df_C.iloc[i, j] = C
            df_theta.iloc[i, j] = theta
            df_mu.iloc[i, j] = mu
            df_sigma.iloc[i, j] = sigma

    return df_mu, df_theta


def add_ttm(df):
    roll_index_df=ExtractDataPostgre.get_delist_date_roll_df(tickers=ConstFutBasic.fut_code_list,start_date='2010-01-01',end_date='2022-08-31')
    list_delist_date_df=ExtractDataPostgre.get_list_date_and_delist_date_of_instrument()
    list_delist_date_df=list_delist_date_df[['ts_code','delist_date']]
    df=pd.merge(left=roll_index_df,right=list_delist_date_df,how='left',left_on='current_main_instrumentid',right_on='ts_code')
    df=df[['main_contract_code','tradingday','delist_date']]




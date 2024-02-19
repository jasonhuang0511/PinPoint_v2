import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kendalltau, pearsonr
import datetime

from extract_data import extract_data_from_postgre as ed
from constant_file.future_data import basic_information as cf
from data_process import future as dp
from factor import momentum as factor


def descriptive_one_case(df):
    df1 = df.dropna()
    result = pd.Series(data=np.nan,
                       index=['stationary'])
    result['stationary'] = adfuller(df1['ret'])[1]

    return result


def process_df(data, window_list, ticker, ret_df_index=True):
    result = pd.DataFrame()
    for j in range(len(window_list)):
        window = window_list[j]
        try:
            print("start: " + window_list[j])
            unit = window[-1]
            scale = int(window[:-1])
            if unit == 'H':
                unit_row = 60
            else:
                unit_row = 1
            df = data.copy()
            df = df.fillna(0)
            print('load data')
            if window == '1T' or window == '1D':
                pass
            else:
                if ret_df_index:
                    df['ret'] = df['ret'].apply(lambda x: x + 1)
                    df['ret_cum'] = df['ret'].rolling(unit_row * scale).apply(np.nanprod)
                    df['ret'] = df['ret'].apply(lambda x: x - 1)
                    df['ret_cum'] = df['ret_cum'].apply(lambda x: x - 1)
                    df = pd.DataFrame(df[::unit_row * scale], index=df.index[::unit_row * scale])
                    df = df[['Code', 'Trade_DT', 'ret_cum']]
                    df.columns = ['Code', 'Trade_DT', 'ret']
                else:
                    df = pd.DataFrame(df[::unit_row * scale], index=df.index[::unit_row * scale])
            df.index = range(len(df))
            print('resample succeed')
            if unit == 'T' and scale <= 10:
                df = dp.delete_open_close_data(df, ticker, open_filter=5, close_filter=5)
            print('delete open close data succeed')

            print('momentum data ok')
            result_one_series = descriptive_one_case(df)
            print('calculating one series')
            result[window] = result_one_series
            print("end: " + window_list[j])
        except Exception as e:
            result_one_series = pd.Series(data=np.nan,
                                          index=['stationary'])
            result[window] = result_one_series

    return result


def descriptive_one_pair_ticker_term_new(ticker):
    # window_list = tickers_para_dict[tickers]['window_list']
    # start_date = tickers_para_dict[tickers]['start_date']
    # end_date = tickers_para_dict[tickers]['end_date']
    # key_word = tickers_para_dict[tickers]['key_word']
    # index = tickers_para_dict[tickers]['index']

    # pct

    # spread
    list_date1 = cf.code_listing_date[ticker]
    end_date = '2022-08-01'
    window_list = ['1T', '2T', '5T', '10T', '20T', '30T', '1H', '2H', '4H', '1D', '2D', '3D', '5D']
    window_list_minute = [i for i in window_list if i[-1] != 'D']
    window_list_daily = [i for i in window_list if i[-1] == 'D']
    index = 1
    key_word = 'close'
    if len(window_list_minute) > 0:
        start_date = '2019-08-01'
        list_date_datetime1 = datetime.datetime.strptime(list_date1, '%Y-%m-%d')
        delta_days = list_date_datetime1 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime = list_date_datetime1.replace(year=list_date_datetime1.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime, "%Y-%m-%d")

        df1 = ed.get_syn_con_ts(tickers=ticker, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='min', index=1, ret_index=False)
        df1.columns = ['Code', 'Trade_DT', 'ret']
        s1 = start_date

        df2 = ed.get_syn_con_ts(tickers=ticker, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='min', index=2, ret_index=False)
        df2.columns = ['Code', 'Trade_DT', 'ret']
        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] - df['ret_y']
        df['Code'] = ticker
        df = df[['Code', 'Trade_DT', 'ret']]

        result_min = process_df(df, window_list_minute, ticker, ret_df_index=False)

        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] / df['ret_y']
        df['Code'] = ticker
        df = df[['Code', 'Trade_DT', 'ret']]

        result_min_ratio = process_df(df, window_list_minute, ticker, ret_df_index=False)
        print("load min data")
        s2 = start_date
    else:
        result_min = pd.DataFrame()
        result_min_ratio = pd.DataFrame()
        s1 = ''
        s2 = ''

    if len(window_list_daily) > 0:
        key_word = 'close'
        start_date = '2017-08-01'

        list_date_datetime1 = datetime.datetime.strptime(list_date1, '%Y-%m-%d')
        delta_days = list_date_datetime1 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime = list_date_datetime1.replace(year=list_date_datetime1.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime, "%Y-%m-%d")

        df1 = ed.get_syn_con_ts(tickers=ticker, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='D', index=1, ret_index=False)
        df1.columns = ['Code', 'Trade_DT', 'ret']
        s3 = start_date

        df2 = ed.get_syn_con_ts(tickers=ticker, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='D', index=2, ret_index=False)
        df2.columns = ['Code', 'Trade_DT', 'ret']
        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] - df['ret_y']
        df['Code'] = ticker
        df = df[['Code', 'Trade_DT', 'ret']]
        result_daily = process_df(df, window_list_daily, ticker, ret_df_index=False)

        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] / df['ret_y']
        df['Code'] = ticker
        df = df[['Code', 'Trade_DT', 'ret']]
        result_daily_ratio = process_df(df, window_list_daily, ticker, ret_df_index=False)
        print("load daily data")
        s4 = start_date
    else:
        result_daily = pd.DataFrame()
        result_daily_ratio = pd.DataFrame()
        s3 = ''
        s4 = ''
    result_spread = result_min.T.append(result_daily.T).T
    str_time_spread = 'min' + s1 + '_' + s2 + '_' + 'daily' + s3 + '_' + s4
    result_ratio = result_min_ratio.T.append(result_daily_ratio.T).T

    return result_spread, str_time_spread, result_ratio


def spread_test():
    file_path2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\term_spread\\stationary\\'
    file_path3 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\term_ratio\\stationary\\'

    code_list = cf.code_list
    for i in range(30,len(code_list)):
        ticker = code_list[i]
        result_spread, str_time_spread, result_ratio = descriptive_one_pair_ticker_term_new(
            ticker)

        file_location2 = file_path2 + ticker + str_time_spread + '.csv'
        file_location3 = file_path3 + ticker + str_time_spread + '.csv'

        result_spread.to_csv(file_location2)
        result_ratio.to_csv(file_location3)
        print(file_location2 + 'is ok')
        print(file_location3 + 'is ok')


if __name__ == '__main__':
    spread_test()

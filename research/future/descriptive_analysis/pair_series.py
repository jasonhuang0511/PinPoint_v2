import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kendalltau, pearsonr
import datetime

from extract_data import extract_data_from_postgre as ed
from constant_file.future_data import basic_information as cf
from data_process import future as dp
from factor import momentum as factor


# from my_stat.time_series import time_series_distance as tsd


def descriptive_one_case(df, price_index=False):
    df1 = df.dropna()
    result = pd.Series(data=np.nan,
                       index=['stationary', 'mean', 'std', 'skew', 'mean_minus_median', 'P_1sigma', 'P_2sigma',
                              'P_3sigma', 'P_positive', 'IC_5_1', 'IC_10_1', 'IC_15_1', 'IC_20_1', 'RANK_IC_5_1',
                              'RANK_IC_10_1', 'RANK_IC_15_1', 'RANK_IC_20_1', 'IC_p_5_1', 'IC_p_10_1', 'IC_p_15_1',
                              'IC_p_20_1', 'RANK_IC_p_5_1',
                              'RANK_IC_p_10_1', 'RANK_IC_p_15_1', 'RANK_IC_p_20_1'])
    if price_index:
        result['stationary'] = adfuller(df1['p'])[1]
        result['mean'] = df1['p'].mean()
        result['std'] = df1['p'].std()
        result['skew'] = df1['p'].skew()
        result['mean_minus_median'] = df1.iloc[:, 2].mean() - df1.iloc[:, 2].median()
        result['P_1sigma'] = (len(df1[df1['p'] > result['mean'] + 1 * result['std']]) + len(
            df1[df1['p'] < result['mean'] - 1 * result['std']])) / len(df1)
        result['P_2sigma'] = (len(df1[df1['p'] > result['mean'] + 2 * result['std']]) + len(
            df1[df1['p'] < result['mean'] - 2 * result['std']])) / len(df1)
        result['P_3sigma'] = (len(df1[df1['p'] > result['mean'] + 3 * result['std']]) + len(
            df1[df1['p'] < result['mean'] - 3 * result['std']])) / len(df1)
        result['P_positive'] = len(df1[df1['p'] > 0]) / len(df1)
    else:

        result['stationary'] = adfuller(df1['ret'])[1]
        result['mean'] = df1['ret'].mean()
        result['std'] = df1['ret'].std()
        result['skew'] = df1['ret'].skew()
        result['mean_minus_median'] = df1.iloc[:, 2].mean() - df1.iloc[:, 2].median()
        result['P_1sigma'] = (len(df1[df1['ret'] > result['mean'] + 1 * result['std']]) + len(
            df1[df1['ret'] < result['mean'] - 1 * result['std']])) / len(df1)
        result['P_2sigma'] = (len(df1[df1['ret'] > result['mean'] + 2 * result['std']]) + len(
            df1[df1['ret'] < result['mean'] - 2 * result['std']])) / len(df1)
        result['P_3sigma'] = (len(df1[df1['ret'] > result['mean'] + 3 * result['std']]) + len(
            df1[df1['ret'] < result['mean'] - 3 * result['std']])) / len(df1)
        result['P_positive'] = len(df1[df1['ret'] > 0]) / len(df1)

    r = pearsonr(df1['ret'][5:], df1['ret_5'][4:-1])
    result['IC_5_1'] = r[0]
    result['IC_p_5_1'] = r[1]

    r = pearsonr(df1['ret'][10:], df1['ret_5'][9:-1])
    result['IC_10_1'] = r[0]
    result['IC_p_10_1'] = r[1]

    r = pearsonr(df1['ret'][15:], df1['ret_5'][14:-1])
    result['IC_15_1'] = r[0]
    result['IC_p_15_1'] = r[1]

    r = pearsonr(df1['ret'][20:], df1['ret_5'][19:-1])
    result['IC_20_1'] = r[0]
    result['IC_p_20_1'] = r[1]

    # kendall
    r = kendalltau(df1['ret'][5:], df1['ret_5'][4:-1])
    result['RANK_IC_5_1'] = r[0]
    result['RANK_IC_p_5_1'] = r[1]

    r = kendalltau(df1['ret'][10:], df1['ret_5'][9:-1])
    result['RANK_IC_10_1'] = r[0]
    result['RANK_IC_p_10_1'] = r[1]

    r = kendalltau(df1['ret'][15:], df1['ret_5'][14:-1])
    result['RANK_IC_15_1'] = r[0]
    result['RANK_IC_p_15_1'] = r[1]

    r = kendalltau(df1['ret'][20:], df1['ret_5'][19:-1])
    result['RANK_IC_20_1'] = r[0]
    result['RANK_IC_p_20_1'] = r[1]

    return result


def process_df(data, window_list, ticker1, ticker2, ret_df_index=True):
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
                df = dp.delete_open_close_data(df, ticker1, open_filter=5, close_filter=5)
                df = dp.delete_open_close_data(df, ticker2, open_filter=5, close_filter=5)
            print('delete open close data succeed')
            if ret_df_index:
                df = factor.calculate_momentum_matrix(df)
            else:
                df1 = factor.calculate_non_return_momentum_matrix(df)
                df1['p'] = df['ret']
                df = df1.copy()
            print('momentum data ok')
            if ret_df_index:
                result_one_series = descriptive_one_case(df)
            else:
                result_one_series = descriptive_one_case(df, price_index=True)
            print('calculating one series')
            result[window] = result_one_series
            print("end: " + window_list[j])
        except Exception as e:
            result_one_series = pd.Series(data=np.nan,
                                          index=['stationary', 'mean', 'std', 'skew', 'mean_minus_median',
                                                 'P_1sigma',
                                                 'P_2sigma',
                                                 'P_3sigma', 'P_positive', 'IC_5_1', 'IC_10_1', 'IC_15_1',
                                                 'IC_20_1',
                                                 'RANK_IC_5_1',
                                                 'RANK_IC_10_1', 'RANK_IC_15_1', 'RANK_IC_20_1', 'IC_p_5_1',
                                                 'IC_p_10_1',
                                                 'IC_p_15_1',
                                                 'IC_p_20_1', 'RANK_IC_p_5_1',
                                                 'RANK_IC_p_10_1', 'RANK_IC_p_15_1', 'RANK_IC_p_20_1'])
            result[window] = result_one_series

    return result


def descriptive_one_pair_ticker_new(ticker1, ticker2):
    # window_list = tickers_para_dict[tickers]['window_list']
    # start_date = tickers_para_dict[tickers]['start_date']
    # end_date = tickers_para_dict[tickers]['end_date']
    # key_word = tickers_para_dict[tickers]['key_word']
    # index = tickers_para_dict[tickers]['index']

    # pct
    list_date1 = cf.code_listing_date[ticker1]
    list_date2 = cf.code_listing_date[ticker2]
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

        df1 = ed.get_syn_con_ts(tickers=ticker1, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='min', index=index, ret_index=True)
        df1.columns = ['Code', 'Trade_DT', 'ret']
        s1 = start_date
        start_date = '2019-08-01'
        list_date_datetime2 = datetime.datetime.strptime(list_date2, '%Y-%m-%d')
        delta_days = list_date_datetime2 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime = list_date_datetime2.replace(year=list_date_datetime1.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime, "%Y-%m-%d")

        df2 = ed.get_syn_con_ts(tickers=ticker2, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='min', index=index, ret_index=True)
        df2.columns = ['Code', 'Trade_DT', 'ret']
        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] - df['ret_y']
        df['Code'] = ticker1 + '_' + ticker2
        df = df[['Code', 'Trade_DT', 'ret']]

        result_min = process_df(df, window_list_minute, ticker1, ticker2)
        print("load min data")
        s2 = start_date
    else:
        result_min = pd.DataFrame()
        s1 = ''
        s2 = ''

    if len(window_list_daily) > 0:
        key_word = 'settle'
        start_date = '2017-08-01'

        list_date_datetime1 = datetime.datetime.strptime(list_date1, '%Y-%m-%d')
        delta_days = list_date_datetime1 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime = list_date_datetime1.replace(year=list_date_datetime1.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime, "%Y-%m-%d")

        df1 = ed.get_syn_con_ts(tickers=ticker1, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='D', index=index, ret_index=True)
        df1.columns = ['Code', 'Trade_DT', 'ret']
        s3 = start_date

        start_date = '2017-08-01'
        list_date_datetime2 = datetime.datetime.strptime(list_date2, '%Y-%m-%d')
        delta_days = list_date_datetime2 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime2 = list_date_datetime1.replace(year=list_date_datetime2.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime2, "%Y-%m-%d")

        df2 = ed.get_syn_con_ts(tickers=ticker2, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='D', index=index, ret_index=True)
        df2.columns = ['Code', 'Trade_DT', 'ret']
        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] - df['ret_y']
        df['Code'] = ticker1 + '_' + ticker2
        df = df[['Code', 'Trade_DT', 'ret']]
        result_daily = process_df(df, window_list_daily, ticker1, ticker2)
        print("load daily data")
        s4 = start_date
    else:
        result_daily = pd.DataFrame()
        s3 = ''
        s4 = ''
    result_pct = result_min.T.append(result_daily.T).T
    str_time_pct = 'min' + s1 + '_' + s2 + '_' + 'daily' + s3 + '_' + s4

    # spread
    list_date1 = cf.code_listing_date[ticker1]
    list_date2 = cf.code_listing_date[ticker2]
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

        df1 = ed.get_syn_con_ts(tickers=ticker1, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='min', index=index, ret_index=False)
        df1.columns = ['Code', 'Trade_DT', 'ret']
        s1 = start_date
        start_date = '2019-08-01'
        list_date_datetime2 = datetime.datetime.strptime(list_date2, '%Y-%m-%d')
        delta_days = list_date_datetime2 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime = list_date_datetime2.replace(year=list_date_datetime1.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime, "%Y-%m-%d")

        df2 = ed.get_syn_con_ts(tickers=ticker2, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='min', index=index, ret_index=False)
        df2.columns = ['Code', 'Trade_DT', 'ret']
        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] - df['ret_y']
        df['Code'] = ticker1 + '_' + ticker2
        df = df[['Code', 'Trade_DT', 'ret']]

        result_min = process_df(df, window_list_minute, ticker1, ticker2, ret_df_index=False)

        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] / df['ret_y']
        df['Code'] = ticker1 + '_' + ticker2
        df = df[['Code', 'Trade_DT', 'ret']]

        result_min_ratio = process_df(df, window_list_minute, ticker1, ticker2, ret_df_index=False)
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

        df1 = ed.get_syn_con_ts(tickers=ticker1, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='D', index=index, ret_index=False)
        df1.columns = ['Code', 'Trade_DT', 'ret']
        s3 = start_date

        start_date = '2017-08-01'
        list_date_datetime2 = datetime.datetime.strptime(list_date2, '%Y-%m-%d')
        delta_days = list_date_datetime2 - datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if delta_days.days + 365 > 0:
            list_date_datetime2 = list_date_datetime1.replace(year=list_date_datetime2.year + 1)
            start_date = datetime.datetime.strftime(list_date_datetime2, "%Y-%m-%d")

        df2 = ed.get_syn_con_ts(tickers=ticker2, start_date=start_date, end_date=end_date,
                                key_word=key_word, freq='D', index=index, ret_index=False)
        df2.columns = ['Code', 'Trade_DT', 'ret']
        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] - df['ret_y']
        df['Code'] = ticker1 + '_' + ticker2
        df = df[['Code', 'Trade_DT', 'ret']]
        result_daily = process_df(df, window_list_daily, ticker1, ticker2, ret_df_index=False)

        df = pd.merge(left=df1, right=df2, how='inner', on='Trade_DT')
        df['ret'] = df['ret_x'] / df['ret_y']
        df['Code'] = ticker1 + '_' + ticker2
        df = df[['Code', 'Trade_DT', 'ret']]
        result_daily_ratio = process_df(df, window_list_daily, ticker1, ticker2, ret_df_index=False)
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

    return result_pct, str_time_pct, result_spread, str_time_spread, result_ratio


def spread_test():
    file_location = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\pair_instrument\\DTW\\all_distance.csv'
    file_path1 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\pair_instrument\\spread_return\\'
    file_path2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\pair_instrument\\spread_price\\'
    file_path3 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\pair_instrument\\spread_ratio\\'
    dis_df = pd.read_csv(file_location).iloc[:, 1:]
    dis_df = dis_df[dis_df['ticker1_start_date'] == '2019-08-01']
    dis_df = dis_df[dis_df['ticker2_start_date'] == '2019-08-01']
    dis_df = dis_df.sort_values(['DTW_disistance'])
    dis_df.index = range(len(dis_df))
    dis_df = dis_df.iloc[:100, :]
    dis_df.index = range(len(dis_df))

    # ratio = pd.DataFrame(cf.cm_cn_product_ratio_dict).T
    # spread = pd.DataFrame(cf.cm_cn_product_spread_dict).T
    # dis_df2 = ratio.append(spread)
    # dis_df2.index = range(len(dis_df2))
    # dis_df2.columns = ['ticker1', 'ticker2']
    # code_mapping={key.split('.')[0]:key for key in cf.code_list}
    # for i in range(5,10):
    #     ticker1 =code_mapping[ dis_df2['ticker1'][i]]
    #     ticker2 = code_mapping[dis_df2['ticker2'][i]]
    #
    #     a = dis_df[dis_df['ticker1'] == ticker1]
    #     a = a[a['ticker2'] == ticker2]
    #     b = dis_df[dis_df['ticker2'] == ticker1]
    #     b = b[b['ticker1'] == ticker2]
    #     if len(a) == 0 and len(b) == 0:
    #         result_pct, str_time_pct, result_spread, str_time_spread, result_ratio = descriptive_one_pair_ticker_new(
    #             ticker1, ticker2)
    #         file_location1 = file_path1 + ticker1 + '_' + ticker2 + str_time_pct + '.csv'
    #         file_location2 = file_path2 + ticker1 + '_' + ticker2 + str_time_spread + '.csv'
    #         file_location3 = file_path3 + ticker1 + '_' + ticker2 + str_time_spread + '.csv'
    #         result_pct.to_csv(file_location1)
    #         result_spread.to_csv(file_location2)
    #         result_ratio.to_csv(file_location3)
    #         print(file_location1 + 'is ok')
    #         print(file_location2 + 'is ok')
    #         print(file_location3 + 'is ok')

    for i in range(len(dis_df)):
        ticker1 = dis_df['ticker1'][i]
        ticker2 = dis_df['ticker2'][i]
        result_pct, str_time_pct, result_spread, str_time_spread, result_ratio = descriptive_one_pair_ticker_new(
            ticker1, ticker2)
        file_location1 = file_path1 + ticker1 + '_' + ticker2 + str_time_pct + '.csv'
        file_location2 = file_path2 + ticker1 + '_' + ticker2 + str_time_spread + '.csv'
        file_location3 = file_path3 + ticker1 + '_' + ticker2 + str_time_spread + '.csv'
        result_pct.to_csv(file_location1)
        result_spread.to_csv(file_location2)
        result_ratio.to_csv(file_location3)
        print(file_location1 + 'is ok')
        print(file_location2 + 'is ok')
        print(file_location3 + 'is ok')


# def dtw_two_series():
#     code_list = cf.code_list
#     file_path = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\pair_instrument\\DTW\\all_distance.csv'
#     end_date = '2022-08-01'
#
#     index = 1
#     key_word = 'close'
#     data_all = pd.DataFrame()
#     for i in range(len(code_list) - 1):
#         ticker1 = code_list[i]
#         start_date1 = dp.adjust_start_date(start_date='2019-08-01', tickers=ticker1)
#         df1 = ed.get_syn_con_ts(tickers=ticker1, start_date=start_date1, end_date=end_date,
#                                 key_word=key_word,
#                                 freq='D', index=index, ret_index=True)
#         df1.columns = ['Code', 'Trade_DT', 'ret']
#         data = pd.DataFrame()
#         for j in range(i + 1, len(code_list)):
#             ticker2 = code_list[j]
#             start_date2 = dp.adjust_start_date(start_date='2019-08-01', tickers=ticker2)
#             df2 = ed.get_syn_con_ts(tickers=ticker2, start_date=start_date2, end_date=end_date,
#                                     key_word=key_word,
#                                     freq='D', index=index, ret_index=True)
#             df2.columns = ['Code', 'Trade_DT', 'ret']
#             data.loc[j - i - 1, 'ticker1'] = ticker1
#             data.loc[j - i - 1, 'ticker2'] = ticker2
#             data.loc[j - i - 1, 'ticker1_start_date'] = start_date1
#             data.loc[j - i - 1, 'ticker2_start_date'] = start_date2
#             data.loc[j - i - 1, 'DTW_disistance'] = tsd.dtw_distance(x=df1.iloc[:, -1], y=df2.iloc[:, -1])
#             print(str(i) + ',' + str(j) + ' :' + ticker1 + ' ' + ticker2 + ' is ok')
#         data_all = data_all.append(data)
#     data_all.to_csv(file_path)
if __name__ == '__main__':
    spread_test()

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
from scipy.stats import shapiro

import model.constants.genetic as ConstGenetic
import data.LoadLocalData.load_local_data as ExtractLocalData
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import model.constants.futures as ConstFutBasic

save_path = 'C:\\Users\\jason.huang\\research\\feature_engineering\\'

tickers_list = ConstFutBasic.fut_code_list
start_date = '2015-01-01'
end_date = '2022-10-30'

close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq='D')
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())


# pct = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
# pct = pct.div(pct.shift(1)) - 1


def roll_box_cox(data):
    return boxcox(data)[0][-1]


start_date = '2019-01-01'
end_date = '2022-09-30'
data = close.copy()
name='close'

def feature_engineering(data, start_date, end_date, name):
    indicator_result_all = pd.DataFrame(
        columns=['tickers', 'normalize_method', 'normed_method', 'outlier_method', 'period', 'mean', 'median', 'range',
                 'IQR', 'count_2sigma', 'normal_test_p_value'])

    data_pivot = data.pivot_table(index='Trade_DT', columns='Fut_code', values=data.columns[2])

    # fill na
    data_pivot = data_pivot[data_pivot.bfill().notnull() & data_pivot.ffill().notnull()] = data_pivot.fillna(
        method='ffill')

    roll_num_list = [20, 40, 60, 90, 180, 256, 512, 1024]
    normed_method_list = ['minmax', 'ts_zscore', 'None']
    normalize_method_list = ['log', 'sqrt', 'empirical_cdf', 'None']
    outlier_method_list = ['MAD', '3sigma', 'None']

    for tickers in tickers_list:
        indicator_one_ticker = pd.DataFrame(
            columns=['tickers', 'normalize_method', 'normed_method', 'outlier_method', 'period', 'mean', 'median',
                     'range', 'IQR', 'count_2sigma', 'normal_test_p_value'])
        df_select = pd.DataFrame(data_pivot[tickers])
        df_select.index = pd.to_datetime(df_select.index)
        index_start = np.where(df_select.index >= pd.to_datetime(start_date))[0][0]
        index_end = np.where(df_select.index > pd.to_datetime(end_date))[0][0]
        for outlier_method in outlier_method_list:
            df1 = pd.DataFrame(
                columns=['tickers', 'normalize_method', 'normed_method', 'outlier_method', 'period', 'mean', 'median',
                         'range', 'IQR', 'count_2sigma', 'normal_test_p_value'])
            for normalize_method in normalize_method_list:
                df2 = pd.DataFrame(
                    columns=['tickers', 'normalize_method', 'normed_method', 'outlier_method', 'period', 'mean',
                             'median', 'range', 'IQR', 'count_2sigma', 'normal_test_p_value'])

                for normed_method in normed_method_list:
                    df3 = pd.DataFrame(
                        columns=['tickers', 'normalize_method', 'normed_method', 'outlier_method', 'period', 'mean',
                                 'median', 'range', 'IQR', 'count_2sigma', 'normal_test_p_value'])

                    for roll_num in roll_num_list:
                        print(f"{tickers}, {outlier_method},{normalize_method},{normed_method},{roll_num} starts")
                        df4 = pd.DataFrame(
                            columns=['tickers', 'normalize_method', 'normed_method', 'outlier_method', 'period', 'mean',
                                     'median', 'range', 'IQR', 'count_2sigma', 'normal_test_p_value'])
                        df_roll = df_select.copy().reset_index(drop=True)
                        df_roll_processed = df_roll.copy()
                        if outlier_method == 'MAD':
                            for i in range(1, len(df_roll)):
                                if i < roll_num:
                                    df = df_roll.iloc[:i, :]
                                else:
                                    df = df_roll.iloc[(i - roll_num):i, :]
                                median_value = df.iloc[:, 0].median()
                                mad = df.sub(df.median()).abs().iloc[:, 0].median()
                                if df.iloc[-1, 0] < median_value - 5 * mad:
                                    df_roll_processed.iloc[i, 0] = median_value - 5 * mad
                                if df.iloc[-1, 0] > median_value + 5 * mad:
                                    df_roll_processed.iloc[i, 0] = median_value + 5 * mad
                        elif outlier_method == '3sigma':
                            for i in range(1, len(df_roll)):
                                if i < roll_num:
                                    df = df_roll.iloc[:i, :]
                                else:
                                    df = df_roll.iloc[(i - roll_num):i, :]
                                mean_value = df.iloc[:, 0].mean()
                                std_value = df.iloc[:, 0].std()
                                if df.iloc[-1, 0] < mean_value - 3 * std_value:
                                    df_roll_processed.iloc[i, 0] = mean_value - 3 * std_value
                                if df.iloc[-1, 0] > mean_value + 3 * std_value:
                                    df_roll_processed.iloc[i, 0] = mean_value + 3 * std_value
                        else:
                            pass
                        df_roll = df_roll_processed.copy()
                        df_roll_processed = df_roll.copy()
                        #########################################
                        if normalize_method == 'box-cox':
                            try:
                                df_roll_processed = df_roll.rolling(window=roll_num, min_periods=10).apply(roll_box_cox)
                            except:
                                pass
                        elif normalize_method == 'log':
                            try:
                                df_roll_processed = df_roll.applymap(np.log)
                            except:
                                pass
                        elif normalize_method == 'sqrt':
                            try:
                                df_roll_processed = df_roll.applymap(np.sqrt)
                            except:
                                pass
                        elif normalize_method == 'empirical_cdf':
                            for i in range(len(df_roll)):
                                if i < 10:
                                    df_roll_processed.iloc[i, 0] = np.nan
                                else:
                                    if i < roll_num:
                                        df = df_roll.iloc[:i, :]
                                    else:
                                        df = df_roll.iloc[(i - roll_num):i, :]
                                    df_roll_processed.iloc[i, 0] = len(df[df.iloc[:, 0] < df.iloc[-1, 0]]) / len(df)
                        else:
                            pass
                        df_roll = df_roll_processed.copy()
                        df_roll_processed = df_roll.copy()
                        ###################################################################
                        if normed_method == "minmax":
                            df_roll['data'] = (df_roll[df_select.columns] -
                                               df_roll.rolling(window=roll_num, min_periods=1)[
                                                   df_select.columns].apply(
                                                   np.nanmin)) / (df_roll.rolling(window=roll_num, min_periods=1)[
                                                                      df_select.columns].max() -
                                                                  df_roll.rolling(window=roll_num, min_periods=1)[
                                                                      df_select.columns].min())
                        elif normed_method == "ts_zscore":
                            df_roll['data'] = (df_roll[df_select.columns] -
                                               df_roll.rolling(window=roll_num, min_periods=1)[
                                                   df_select.columns].mean()) / \
                                              df_roll.rolling(window=roll_num, min_periods=1)[
                                                  df_select.columns].std()
                        else:
                            df_roll['data'] = df_roll[df_select.columns]
                        df_roll = df_roll.iloc[index_start:index_end, :]
                        # l=len(df_roll)
                        try:
                            df4.loc[0, 'mean'] = df_roll['data'].mean()
                        except:
                            df4.loc[0, 'mean'] = np.nan
                        try:
                            df4.loc[0, 'median'] = df_roll['data'].median()
                        except:
                            df4.loc[0, 'median'] = 0

                        try:
                            df4.loc[0, 'IQR'] = np.nanpercentile(df_roll['data'], 75) - np.nanpercentile(
                                df_roll['data'], 25)
                        except:
                            df4.loc[0, 'IQR'] =np.nan
                        try:
                            df4.loc[0, 'range'] = df_roll['data'].max() - df_roll['data'].min()
                        except:
                            df4.loc[0, 'range'] =np.nan
                        try:
                            df4.loc[0, 'normal_test_p_value'] = shapiro(df_roll.dropna()['data'])[1]
                        except:
                            df4.loc[0, 'normal_test_p_value']=np.nan
                        try:
                            df_roll['z'] = ((df_roll['data'] - df_roll['data'].mean()) / df_roll['data'].std()).abs()
                            df_roll = df_roll[df_roll['z'] > 2]

                            df4.loc[0, 'count_2sigma'] = len(df_roll)
                        except:
                            df4.loc[0, 'count_2sigma'] =np.nan
                        df4['period'] = roll_num

                        print(f"{tickers}, {outlier_method},{normalize_method},{normed_method},{roll_num} is ok")
                        df3 = pd.concat([df3, df4])

                    df3['normed_method'] = normed_method
                    df2 = pd.concat([df2, df3])
                df2['normalize_method'] = normalize_method
                df1 = pd.concat([df1, df2])

            df1['outlier_method'] = outlier_method
            indicator_one_ticker = pd.concat([indicator_one_ticker, df1])
        indicator_one_ticker['tickers'] = tickers
        indicator_result_all = pd.concat([indicator_result_all, indicator_one_ticker])

        indicator_result_all.to_csv(save_path + name + '.csv')
    return indicator_result_all


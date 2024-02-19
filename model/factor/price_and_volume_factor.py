import pandas as pd
import numpy as np
import datetime
from scipy.stats.stats import kendalltau
from scipy.stats.stats import pearsonr

import data.ConstantData.future_basic_information as ConstFutBasic
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.Process.fut_data_process as DataProcessFut


# 计算open interest & close trend
def open_interest_trend(tickers, start_date, end_date):
    main_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date,
                                                             index=1)
    second_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date,
                                                               end_date=end_date, index=2)
    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    close_price = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list, start_date=start_date,
                                                              end_date=end_date,
                                                              key_word='close')
    oi = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='oi')

    close_price['fut_code'] = close_price['ts_code'].map(ExtractDataPostgre.get_code_instrument_mapping())
    oi['fut_code'] = oi['ts_code'].map(ExtractDataPostgre.get_code_instrument_mapping())

    code_list = np.unique(close_price['fut_code'])
    factor_df = pd.DataFrame()
    for i in range(len(code_list)):
        code = code_list[i]
        d1 = close_price[close_price['fut_code'] == code]
        f1 = d1.pivot_table(index='trade_date', columns='ts_code', values='close')
        f1 = f1.diff().applymap(np.sign)

        d2 = oi[oi['fut_code'] == code]
        f2 = d2.pivot_table(index='trade_date', columns='ts_code', values='oi')
        f2 = f2.diff().applymap(np.sign)

        f = f1 * f2

        f = f.fillna(method='backfill', axis=1).reset_index().iloc[:, :2]
        f.columns = ['tradingday', code]
        if i == 0:
            factor_df = f
        else:
            factor_df = pd.merge(left=factor_df, right=f, on='tradingday', how='outer')
    factor_df = factor_df.sort_values('tradingday')
    factor_df.index = factor_df['tradingday']
    factor_df = factor_df[np.sort(factor_df.columns[1:])]
    factor_df = factor_df.rolling(10).sum()
    return factor_df


# 计算收益率momentum矩阵
def calculate_momentum_matrix(df, params=None, multi_code=False, long_format_index=True):
    if params is None:
        params = [5, 10, 15, 20]

    if multi_code:
        if params is not None:
            if type(params) is list:
                param = params[0]
            elif type(params) is int:
                param = params
            else:
                raise TypeError('params should be list or int')
        if long_format_index:
            df = df.pivot_table(values=df.columns[2], index=df.columns[1], columns=df.columns[0])
        df = df.add(1)
        df = df.apply(lambda x: x.rolling(param).apply(np.nanprod))
        df = df.add(-1)


    else:

        if not long_format_index:
            df = pd.DataFrame(df)
            df['Trade_DT'] = df.index
            df = df.melt('Trade_DT')
            df.columns = ['Trade_DT', 'Code', 'ret']
            df = df[['Code', 'Trade_DT', 'ret']]

        df.columns = ['Code', 'Trade_DT', 'ret']
        df['ret'] = df['ret'].apply(lambda x: x + 1)
        for param in params:
            df['ret_' + str(param)] = df['ret'].rolling(param).apply(np.nanprod) - 1
        df['ret'] = df['ret'].apply(lambda x: x - 1)

    return df


# 计算非收益率momentum矩阵
def calculate_non_return_momentum_matrix(data, params=None, multi_code=False, long_format_index=True):
    df = data.copy()
    if params is None:
        params = [5, 10, 15, 20]

    if multi_code:
        if params is not None:
            if type(params) is list:
                param = params[0]
            elif type(params) is int:
                param = params
            else:
                raise TypeError('params should be list or int')
        if long_format_index:
            df = df.pivot_table(values=df.columns[2], index=df.columns[1], columns=df.columns[0])
        df = df.add(1)
        df = df.apply(lambda x: x.rolling(param).apply(np.nanprod))
        df = df.add(-1)


    else:

        if not long_format_index:
            df = pd.DataFrame(df)
            df['Trade_DT'] = df.index
            df = df.melt('Trade_DT')
            df.columns = ['Trade_DT', 'Code', 'ret']
            df = df[['Code', 'Trade_DT', 'ret']]

        df.columns = ['Code', 'Trade_DT', 'ret']
        df['ret'] = df['ret'].diff(1)
        for param in params:
            df['ret_' + str(param)] = df['ret'].rolling(param).apply(np.nansum)
        # df['ret'] = df['ret'].apply(lambda x: x - 1)

    return df


# intraday current main and next main OI correlation
def calculate_intraday_pair_oi_correlation():
    threshold_num = 10
    tickers = ConstFutBasic.fut_code_list
    start_date = '2018-01-01'
    end_date = '2022-08-31'
    roll_df = ExtractDataPostgre.get_delist_date_roll_df(tickers, start_date, end_date)
    roll_df = roll_df.sort_values(['main_contract_code', 'tradingday']).reset_index(drop=True)
    # roll_df = roll_df[['fut_code', 'tradingday', 'main']]
    roll_df.columns = ['code', 'tradingday', 'current_main_instrumentid']

    main_contract_mapping_dict = ExtractDataPostgre.generate_main_contract_mapping()
    roll_df['second_main_instrumentid'] = roll_df['current_main_instrumentid'].map(main_contract_mapping_dict)
    roll_df['third_main_instrumentid'] = roll_df['second_main_instrumentid'].map(main_contract_mapping_dict)

    main_contract_mapping_dict = ExtractDataPostgre.generate_main_contract_mapping(start_year=2018, end_year=2022)
    main_df = pd.DataFrame(main_contract_mapping_dict.items(), columns=['main', 'second'])
    delist_date_df = ExtractDataPostgre.get_list_date_and_delist_date_of_instrument()
    main_df = pd.merge(left=main_df, right=delist_date_df, how='inner', left_on='main', right_on='ts_code')
    main_df = main_df[['main', 'second', 'delist_date']].reset_index(drop=True)
    data_all_correlation = pd.DataFrame()
    data_all_p_value = pd.DataFrame()

    for i in range(len(main_df)):
        tickers_list = [main_df.loc[i, 'main'], main_df.loc[i, 'second']]
        print('start of ' + str(i) + ' / ' + str(len(main_df)) + '   ' + ','.join(tickers_list))

        file_name = '_'.join(tickers_list) + '.csv'
        file_location = 'C:\\Users\\jason.huang\\research\\factor_data\\OI_change\\intraday_OI_correlation\\' + file_name

        main = main_df.loc[i, 'main']
        second = main_df.loc[i, 'second']
        try:
            code = ExtractDataPostgre.get_code_instrument_mapping()[main_df.loc[i, 'main']]
        except Exception as e:
            code = main_df.loc[i, 'main'].split('.')[0][:-4] + '.' + main_df.loc[i, 'main'].split('.')[1]
        try:
            end_date_tickers = datetime.datetime.strftime(main_df.loc[i, 'delist_date'], '%Y-%m-%d')
            start_date_tickers = datetime.datetime.strftime(
                main_df.loc[i, 'delist_date'] - datetime.timedelta(days=120),
                '%Y-%m-%d')
            data = ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers_list, start_date=start_date_tickers,
                                                                end_date=end_date_tickers,
                                                                key_word=['tradingday', 'position'])
            position = ExtractDataPostgre.get_future_minute_data_sm(tickers=tickers_list, start_date=start_date_tickers,
                                                                    end_date=end_date_tickers,
                                                                    key_word=['tradingday', 'position'])
            time_mapping = data.iloc[:, [1, 2]]
            data = data.pivot_table(index='datetime', columns='windcode', values='position')
            data = data.diff().reset_index()
            data = pd.merge(left=data, right=time_mapping, how='left', left_on='datetime', right_on='datetime')
            data = data.drop_duplicates()
            # data['tradingday'] = data['datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
            data.to_csv(file_location)

            tradingday_list = np.sort(np.unique(data['tradingday']))
            result = pd.DataFrame()
            result_p = pd.DataFrame()
            for t in range(len(tradingday_list)):

                print('start of tradingdaylist :   ' + str(t) + ' / ' + str(len(tradingday_list)))
                tradingday = tradingday_list[t]

                result.loc[t, 'main'] = main
                result.loc[t, 'second'] = second
                result.loc[t, 'code'] = code
                result.loc[t, 'tradingday'] = tradingday

                result_p.loc[t, 'main'] = main
                result_p.loc[t, 'second'] = second
                result_p.loc[t, 'code'] = code
                result_p.loc[t, 'tradingday'] = tradingday

                df = data[data['tradingday'] == tradingday].dropna()
                if len(df) < 100:
                    pass
                else:
                    # 正常correlation
                    v1 = np.array(df.iloc[:, 1])
                    v2 = np.array(df.iloc[:, 2])
                    try:
                        result.loc[t, 'ic'], result_p.loc[t, 'ic'] = pearsonr(v1, v2)
                    except Exception as e:
                        result.loc[t, 'ic'] = np.nan
                        result_p.loc[t, 'ic'] = np.nan
                    try:
                        result.loc[t, 'rankic'], result_p.loc[t, 'rankic'] = kendalltau(v1, v2)
                    except Exception as e:
                        result.loc[t, 'rankic'] = np.nan
                        result_p.loc[t, 'rankic'] = np.nan

                    # 根据大小区分
                    df = df.sort_values(df.columns[1]).reset_index(drop=True)
                    df['index_num'] = np.array([1] * 20 + [0] * (len(df) - 40) + [1] * 20)
                    df = df.sort_values(df.columns[2]).reset_index(drop=True)
                    df['index_num'] = df['index_num'] + np.array([1] * 20 + [0] * (len(df) - 40) + [1] * 20)
                    df_small = df[df['index_num'] == 0]
                    df_large = df[df['index_num'] > 0]
                    v1 = np.array(df_small.iloc[:, 1])
                    v2 = np.array(df_small.iloc[:, 2])

                    try:
                        result.loc[t, 'ic_small_num'], result_p.loc[t, 'ic_small_num'] = pearsonr(v1, v2)
                    except Exception as e:
                        result.loc[t, 'ic_small_num'] = np.nan
                        result_p.loc[t, 'ic_small_num'] = np.nan
                    try:
                        result.loc[t, 'rankic_small_num'], result_p.loc[t, 'rankic_small_num'] = kendalltau(v1, v2)
                    except Exception as e:
                        result.loc[t, 'rankic_small_num'] = np.nan
                        result_p.loc[t, 'rankic_small_num'] = np.nan

                    v1 = np.array(df_large.iloc[:, 1])
                    v2 = np.array(df_large.iloc[:, 2])

                    try:
                        result.loc[t, 'ic_large_num'], result_p.loc[t, 'ic_large_num'] = pearsonr(v1, v2)
                    except Exception as e:
                        result.loc[t, 'ic_large_num'] = np.nan
                        result_p.loc[t, 'ic_large_num'] = np.nan
                    try:
                        result.loc[t, 'rankic_large_num'], result_p.loc[t, 'rankic_large_num'] = kendalltau(v1, v2)
                    except Exception as e:
                        result.loc[t, 'rankic_large_num'] = np.nan
                        result_p.loc[t, 'rankic_large_num'] = np.nan
                    # 根据时间区分

                    df_copy = df.copy()
                    df_copy.columns = ['Trade_DT', 'current', 'second', 'tradingday', 'index_num']
                    df_intraday = DataProcessFut.delete_open_close_data(df=df_copy, tickers=code)
                    df_open_close = df_copy[~df_copy['Trade_DT'].isin(df_intraday['Trade_DT'])]

                    v1 = np.array(df_intraday.iloc[:, 1])
                    v2 = np.array(df_intraday.iloc[:, 2])

                    try:
                        result.loc[t, 'ic_intraday'], result_p.loc[t, 'ic_intraday'] = pearsonr(v1, v2)
                    except Exception as e:
                        result.loc[t, 'ic_intraday'] = np.nan
                        result_p.loc[t, 'ic_intraday'] = np.nan
                    try:
                        result.loc[t, 'rankic_intraday'], result_p.loc[t, 'rankic_intraday'] = kendalltau(v1, v2)
                    except Exception as e:
                        result.loc[t, 'rankic_intraday'] = np.nan
                        result_p.loc[t, 'rankic_intraday'] = np.nan

                    v1 = np.array(df_open_close.iloc[:, 1])
                    v2 = np.array(df_open_close.iloc[:, 2])
                    try:
                        result.loc[t, 'ic_open_close'], result_p.loc[t, 'ic_open_close'] = pearsonr(v1, v2)
                    except Exception as e:
                        result.loc[t, 'ic_open_close'] = np.nan
                        result_p.loc[t, 'ic_open_close'] = np.nan
                    try:
                        result.loc[t, 'rankic_open_close'], result_p.loc[t, 'rankic_open_close'] = kendalltau(v1, v2)
                    except Exception as e:
                        result.loc[t, 'rankic_open_close'] = np.nan
                        result_p.loc[t, 'rankic_open_close'] = np.nan

                    # 修复curve区分
                    if t < 10:
                        pass
                    else:
                        data_before_all = pd.DataFrame()
                        for k in range(1, threshold_num + 1):
                            data_before = position[position['tradingday'] == tradingday_list[t - k]].dropna()
                            data_before['hour_minute'] = data_before['datetime'].apply(
                                lambda x: x.hour * 100 + x.minute)
                            data_before = data_before.pivot_table(index='hour_minute', columns='windcode',
                                                                  values='position')
                            if k == 1:
                                data_before_all = data_before
                            else:
                                data_before_all = data_before_all + data_before
                        data_before_all = data_before_all / 10
                        data_before_all = data_before_all.reset_index()

                        data_today = position[position['tradingday'] == tradingday_list[t]].dropna()
                        data_today['hour_minute'] = data_today['datetime'].apply(lambda x: x.hour * 100 + x.minute)
                        data_today = data_today.pivot_table(index='hour_minute', columns='windcode',
                                                            values='position').reset_index()
                        data_today_before = pd.merge(left=data_today, right=data_before_all, how='inner',
                                                     on='hour_minute').dropna()
                        v1 = np.array(data_today_before.iloc[:, 1] - data_today_before.iloc[:, 3])
                        v2 = np.array(data_today_before.iloc[:, 2] - data_today_before.iloc[:, 4])

                        try:
                            result.loc[t, 'ic_oi_shaped'], result_p.loc[t, 'ic_oi_shaped'] = pearsonr(v1, v2)
                        except Exception as e:
                            result.loc[t, 'ic_oi_shaped'] = np.nan
                            result_p.loc[t, 'ic_oi_shaped'] = np.nan
                        try:
                            result.loc[t, 'rankic_oi_shaped'], result_p.loc[t, 'rankic_oi_shaped'] = kendalltau(v1,
                                                                                                                v2)
                        except Exception as e:
                            result.loc[t, 'rankic_oi_shaped'] = np.nan
                            result_p.loc[t, 'rankic_oi_shaped'] = np.nan

            data_all_correlation = pd.concat([data_all_correlation, result])
            data_all_p_value = pd.concat([data_all_p_value, result_p])

            # data = data.groupby('tradingday')[tickers_list].corr().unstack().iloc[:, 1]
            # data = data.reset_index()
            # data.columns = ['tradingday', 'correlation']
            # data['main'] = main_df.loc[i, 'main']
            # data['second'] = main_df.loc[i, 'second']
            # data['code'] = ExtractDataPostgre.get_code_instrument_mapping()[main_df.loc[i, 'main']]
            # data = data[['code', 'main', 'second', 'tradingday', 'correlation']]
            # data_all = pd.concat([data_all, data])
            print(','.join(tickers_list) + ' is ok')
        except Exception as e:
            result = pd.DataFrame(columns=['main', 'second', 'code', 'tradingday', 'ic', 'rankic', 'ic_small_num',
                                           'rankic_small_num', 'ic_large_num', 'rankic_large_num', 'ic_intraday',
                                           'rankic_intraday', 'ic_open_close', 'rankic_open_close', 'ic_oi_shaped',
                                           'rankic_oi_shaped'])
            result.loc[0, 'main'] = main
            result.loc[0, 'second'] = second
            result.loc[0, 'code'] = code
            result.loc[0, 'tradingday'] = main_df.loc[i, 'delist_date']
            result_p = result.copy()
            data_all_correlation = pd.concat([data_all_correlation, result])
            data_all_p_value = pd.concat([data_all_p_value, result_p])

            # data = pd.DataFrame()
            # data.loc[0, 'tradingday'] = main_df.loc[i, 'delist_date']
            # data.loc[0, 'correlation'] = 0
            # data.loc[0, 'main'] = main_df.loc[i, 'main']
            # data.loc[0, 'second'] = main_df.loc[i, 'second']
            # data['code'] = ExtractDataPostgre.get_code_instrument_mapping()[main_df.loc[i, 'main']]
            # data = data[['code', 'main', 'second', 'tradingday', 'correlation']]
            # data_all = pd.concat([data_all, data])
            print(','.join(tickers_list) + ' is not ok')
        data_all_correlation.to_csv("intraday_OI_correlation_all.csv")
        data_all_p_value.to_csv("intraday_OI_correlation_pvalue_all.csv")

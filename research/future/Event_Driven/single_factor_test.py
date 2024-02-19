import pandas as pd
import numpy as np
import os
import datetime

import data.ConstantData.future_basic_information as ConstFutBasic
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import backtest.Backtest_Object as BtObj
import model.factor.price_and_volume_factor as FactorPriceVolume
import research.future.simple_factor_test.cashbasis_warehouse_detrend as detrend
import data.Process.fut_data_process as FutDataProcess
import data.LoadLocalData.load_local_data as LoadLocalData


def create_factor_oi(window, window_yoy, m1, m2):
    fut_code_sort_by_roll_month = ConstFutBasic.fut_code_sort_by_roll_month
    file_path = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_rate\\'
    file_path1 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_diff\\OI_rate_rolling_window' + str(
        window) + '\\'
    file_path2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_diff\\OI_rate_rolling_month' + str(
        window_yoy) + '\\'

    file_path_time = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\date_time\\'
    if not os.path.exists(file_path1):
        os.makedirs(file_path1)
    if not os.path.exists(file_path2):
        os.makedirs(file_path2)

    result_all = pd.DataFrame()

    for file_name in os.listdir(file_path):
        file_location = file_path + file_name
        ticker = file_name[:-4]
        data = pd.read_csv(file_location, index_col=0)
        result1 = pd.DataFrame(index=data.index, columns=data.columns)
        result2 = pd.DataFrame(index=data.index, columns=data.columns)
        for i in range(len(data)):
            for j in range(window, len(data.columns)):
                try:
                    result1.iloc[i, j] = data.iloc[i, j] - np.nanmean(data.iloc[i, (j - window):j])
                except Exception as e:
                    pass
                try:
                    m = np.nan
                    for k in [3, 4, 12]:
                        if ticker in fut_code_sort_by_roll_month[str(k)]:
                            m = int(k)
                    result2.iloc[i, j] = data.iloc[i, j] - np.nanmean(
                        data.iloc[i, [j - m * (k + 1) for k in range(window_yoy)]])
                except Exception as e:
                    pass
        file_location1 = file_path1 + file_name
        file_location2 = file_path2 + file_name
        result1.to_csv(file_location1)
        result2.to_csv(file_location2)

        file_location_time = file_path_time + file_name

        dt = pd.read_csv(file_location_time, index_col=0)
        r1 = result1.copy()
        r2 = result2.copy()
        result = pd.DataFrame()
        k = 0
        for i in range(round(max(len(dt) * 0.6, len(dt) - m1)), len(dt) - m2):
            for j in range(len(dt.columns)):
                result.loc[k, 'code'] = dt.columns[j]
                result.loc[k, 'tradingday'] = dt.iloc[i, j]
                result.loc[k, 'rollingwindow'] = r1.iloc[i, j]
                result.loc[k, 'rollingmonth'] = r2.iloc[i, j]
                k = k + 1
        result_all = result_all.append(result)
        print(file_name + ' is ok')
    result_all = result_all.dropna()
    result_all.to_csv(
        'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal_oi_rate' + str(
            window) + '_' + str(window_yoy) + '_m1' + str(m1) + '_m2' + str(m2) + '.csv')

    fut_code_sort_by_roll_month = ConstFutBasic.fut_code_sort_by_roll_month
    file_path = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_data_matrix\\'
    file_path1 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_diff\\OI_pct_rolling_window' + str(
        window) + '\\'
    file_path2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_diff\\OI_pct_rolling_month' + str(
        window_yoy) + '\\'

    file_path_time = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\date_time\\'
    if not os.path.exists(file_path1):
        os.makedirs(file_path1)
    if not os.path.exists(file_path2):
        os.makedirs(file_path2)

    result_all = pd.DataFrame()

    for file_name in os.listdir(file_path):
        file_location = file_path + file_name
        ticker = file_name[:-4]
        data = pd.read_csv(file_location, index_col=0)
        result1 = pd.DataFrame(index=data.index, columns=data.columns)
        result2 = pd.DataFrame(index=data.index, columns=data.columns)
        for i in range(len(data)):
            for j in range(window, len(data.columns)):
                try:
                    result1.iloc[i, j] = data.iloc[i, j] - np.nanmean(data.iloc[i, (j - window):j])
                except Exception as e:
                    pass
                try:
                    m = np.nan
                    for k in [3, 4, 12]:
                        if ticker in fut_code_sort_by_roll_month[str(k)]:
                            m = int(k)
                    result2.iloc[i, j] = data.iloc[i, j] - np.nanmean(
                        data.iloc[i, [j - m * (k + 1) for k in range(window_yoy)]])
                except Exception as e:
                    pass
        file_location1 = file_path1 + file_name
        file_location2 = file_path2 + file_name
        result1.to_csv(file_location1)
        result2.to_csv(file_location2)

        file_location_time = file_path_time + file_name

        dt = pd.read_csv(file_location_time, index_col=0)
        r1 = result1.copy()
        r2 = result2.copy()
        result = pd.DataFrame()
        k = 0
        for i in range(round(max(len(dt) * 0.6, len(dt) - m1)), len(dt) - m2):
            for j in range(len(dt.columns)):
                result.loc[k, 'code'] = dt.columns[j]
                result.loc[k, 'tradingday'] = dt.iloc[i, j]
                result.loc[k, 'rollingwindow'] = r1.iloc[i, j]
                result.loc[k, 'rollingmonth'] = r2.iloc[i, j]
                k = k + 1
        result_all = result_all.append(result)
        print(file_name + ' is ok')
    result_all = result_all.dropna()

    result_all.to_csv(
        'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal_oi_pct' + str(
            window) + '_' + str(window_yoy) + '_m1' + str(m1) + '_m2' + str(m2) + '.csv')
    result_path1 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal_oi_rate' + str(
        window) + '_' + str(window_yoy) + '_m1' + str(m1) + '_m2' + str(m2) + '.csv'
    result_path2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal_oi_pct' + str(
        window) + '_' + str(window_yoy) + '_m1' + str(m1) + '_m2' + str(m2) + '.csv'
    return result_path1, result_path2


def get_factor_percent_change_df(file_location=None):
    if file_location is None:
        file_location = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal.csv'
    data = pd.read_csv(file_location, index_col=0)
    data['fut_code'] = data['code'].map(ExtractDataPostgre.get_code_instrument_mapping())
    code_list = np.unique(data['fut_code'])
    factor1 = pd.DataFrame()
    factor2 = pd.DataFrame()
    for i in range(len(code_list)):
        code = code_list[i]
        d1 = data[data['fut_code'] == code]
        f1 = d1.pivot_table(index='tradingday', columns='code', values='rollingwindow').fillna(method='backfill',
                                                                                               axis=1).reset_index().iloc[
             :, :2]
        f2 = d1.pivot_table(index='tradingday', columns='code', values='rollingwindow').fillna(method='backfill',
                                                                                               axis=1).reset_index().iloc[
             :, :2]
        f1.columns = ['tradingday', code]
        f2.columns = ['tradingday', code]
        if i == 0:
            factor1 = f1
            factor2 = f2

        else:
            factor1 = pd.merge(left=factor1, right=f1, on='tradingday', how='outer')
            factor2 = pd.merge(left=factor2, right=f2, on='tradingday', how='outer')
    factor1 = factor1.sort_values('tradingday')
    factor1['tradingday'] = [datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])) for x in factor1['tradingday']]
    factor1.index = factor1['tradingday']
    factor1 = factor1[np.sort(factor1.columns[1:])]

    factor2 = factor2.sort_values('tradingday')
    factor2['tradingday'] = [datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])) for x in factor2['tradingday']]
    factor2.index = factor2['tradingday']
    factor2 = factor2[np.sort(factor1.columns[1:])]

    return factor1, factor2


def get_factor_rate_change_df(file_location=None):
    if file_location is None:
        file_location = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal_oi_rate.csv'
    data = pd.read_csv(file_location, index_col=0)
    data['fut_code'] = data['code'].map(ExtractDataPostgre.get_code_instrument_mapping())
    code_list = np.unique(data['fut_code'])
    factor1 = pd.DataFrame()
    factor2 = pd.DataFrame()
    for i in range(len(code_list)):
        code = code_list[i]
        d1 = data[data['fut_code'] == code]
        f1 = d1.pivot_table(index='tradingday', columns='code', values='rollingwindow').fillna(method='backfill',
                                                                                               axis=1).reset_index().iloc[
             :, :2]
        f2 = d1.pivot_table(index='tradingday', columns='code', values='rollingwindow').fillna(method='backfill',
                                                                                               axis=1).reset_index().iloc[
             :, :2]
        f1.columns = ['tradingday', code]
        f2.columns = ['tradingday', code]
        if i == 0:
            factor1 = f1
            factor2 = f2

        else:
            factor1 = pd.merge(left=factor1, right=f1, on='tradingday', how='outer')
            factor2 = pd.merge(left=factor2, right=f2, on='tradingday', how='outer')
    factor1 = factor1.sort_values('tradingday')
    factor1['tradingday'] = [datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])) for x in factor1['tradingday']]
    factor1.index = factor1['tradingday']
    factor1 = factor1[np.sort(factor1.columns[1:])]

    factor2 = factor2.sort_values('tradingday')
    factor2['tradingday'] = [datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])) for x in factor2['tradingday']]
    factor2.index = factor2['tradingday']
    factor2 = factor2[np.sort(factor1.columns[1:])]

    return factor1, factor2


def get_spread_df(tickers, start_date, end_date, roll_df):
    main_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date,
                                                             index=1)
    second_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date,
                                                               end_date=end_date, index=2)
    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    df = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='close', as_matrix=True)
    df = df.T.diff().shift(-1).T
    df = ExtractDataPostgre.combine_to_con_fut_code_ts(df=df, roll_df=roll_df)

    return df


def get_trade_price_df(tickers, start_date, end_date, roll_df):
    main_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date,
                                                             index=1)
    second_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date,
                                                               end_date=end_date, index=2)
    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    df = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='settle', as_matrix=True)
    df = df.T.diff().shift(-1).T
    df = ExtractDataPostgre.combine_to_con_fut_code_ts(df=df, roll_df=roll_df)
    return df


def get_vwap_df(tickers, start_date, end_date, roll_df):
    main_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date,
                                                             index=1)
    second_contract_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date,
                                                               end_date=end_date, index=2)
    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    df = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                     key_word='settle', as_matrix=True)
    for j in range(len(df.columns) - 1):
        df[df.columns[j]] = df[df.columns[j]] + df[df.columns[j + 1]]
    df = ExtractDataPostgre.combine_to_con_fut_code_ts(df=df, roll_df=roll_df)
    return df


def format_factor_df(factor, price_df):
    result = pd.DataFrame(data=0, index=price_df.index, columns=price_df.columns).reset_index(price_df.index.name).melt(
        price_df.index.name)
    result.columns = ['tradingday', 'code', 'value']
    f = factor.reset_index(factor.index.name).melt(factor.index.name)
    f.columns = ['tradingday', 'code', 'factor']
    result = pd.merge(left=result, right=f, left_on=[result.columns[0], result.columns[1]],
                      right_on=[f.columns[0], f.columns[1]], how='left')
    result['factor'] = result['factor'].replace(np.nan, np.inf)
    result = result.pivot_table(index='tradingday', columns='code', values='factor')
    result = result.replace(np.inf, np.nan)
    return result


def get_trading_vol(tickers, start_date, end_date):
    vol1 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers, start_date=start_date, end_date=end_date, key_word='vol',
                                             index=1)
    vol1['vol'] = vol1['vol'].fillna(0)
    vol1 = vol1.dropna()
    vol2 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers, start_date=start_date, end_date=end_date, key_word='vol',
                                             index=2)
    vol2['vol'] = vol2['vol'].fillna(0)
    vol2 = vol2.dropna()

    def remove_digit(str1):
        return ''.join([i for i in str1 if not i.isdigit()])

    vol1['Code'] = vol1['Code'].apply(remove_digit)
    vol2['Code'] = vol2['Code'].apply(remove_digit)

    vol1 = vol1.pivot_table(index='Trade_DT', columns='Code', values='vol')
    vol1 = vol1.fillna(0)
    vol2 = vol2.pivot_table(index='Trade_DT', columns='Code', values='vol')
    vol2 = vol2.fillna(0)

    return vol1, vol2


def get_trading_schedule(vol1, vol2, weight_df, param=None):
    if param is None:
        param = [ConstFutBasic.trading_schedule_rolling_param, ConstFutBasic.trading_schedule_vol_multiplier]
    vol1 = vol1.rolling(param[0]).mean()
    vol2 = vol2.rolling(param[0]).mean()
    vol = pd.concat([vol1, vol2]).min(level=0).iloc[1:, :]
    trading_schedule_df = vol.multiply(param[1]).add(weight_df.abs().multiply(-1)).applymap(lambda x: 1 if x > 0 else 0)
    return trading_schedule_df


def trading_scheduled_weight(tickers, start_date, end_date, weight_df, trading_schedule_df):
    roll_index_df = ExtractDataPostgre.get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date,
                                                          key_word=['roll_next_day'], index=None)
    roll_index_df = roll_index_df.sort_values(['main_contract_code', 'tradingday']).reset_index(drop=True)
    roll_index_df.loc[roll_index_df[roll_index_df['roll_next_day'] == 1].index + 1, 'roll_next_day'] = -1
    roll_index_df = roll_index_df[roll_index_df['roll_next_day'] == -1].reset_index(drop=True)
    roll_index_df['roll_next_day'] = 0
    roll_index_df = roll_index_df.pivot_table(index='tradingday', columns='main_contract_code',
                                              values='roll_next_day').fillna(1)

    # schedule=0 非换月日 不交易
    trading_schedule_df1 = trading_schedule_df.replace(0, np.nan)
    trading_schedule_df1.iloc[0, :] = 0
    weight_df1 = (trading_schedule_df1 * weight_df).fillna(method='ffill')

    # schedule=0 + 当天换月 仓位=0
    # trading_schedule_df2 = trading_schedule_df.replace(0, -1)
    roll_index_df_all = pd.DataFrame(data=1, index=trading_schedule_df.index, columns=trading_schedule_df.columns)
    roll_index_df_all.loc[roll_index_df_all.index.isin(roll_index_df.index), :] = roll_index_df

    weight_df2 = weight_df1 * ((trading_schedule_df + roll_index_df_all).applymap(np.sign))

    return weight_df2


# def get_weight_df(signal_df, factor_oi, cb_diff, cb_theta, wh_diff, wh_theta):
#     result = signal_df.copy()
#     for i in range(len(signal_df)):
#         for j in range(len(signal_df.columns)):
#             if np.isfinite(signal_df.iloc[i, j]) and not pd.isna(signal_df.iloc[i, j]):
#                 if signal_df.iloc[i, j] > 0:
#                     if not pd.isna(cb_diff.iloc[i, j]):
#                         a = np.sign(cb_diff.iloc[i, j])
#                     else:
#                         a = 0
#                     if not pd.isna(cb_theta.iloc[i, j]):
#                         b = np.sign(cb_theta.iloc[i, j])
#                     else:
#                         b = 0
#                     if not pd.isna(wh_diff.iloc[i, j]):
#                         c = np.sign(wh_diff.iloc[i, j])
#                     else:
#                         c = 0
#                     if not pd.isna(wh_theta.iloc[i, j]):
#                         d = np.sign(wh_theta.iloc[i, j])
#                     else:
#                         d = 0
#
#                     result.iloc[i, j] = np.sign(np.sign(factor_oi.iloc[i, j]) * (-1) + a + b - c - d)
#
#                 else:
#                     result.iloc[i, j] = 0
#             else:
#                 result.iloc[i, j] = 0
#     return result

def get_weight_df(factor):
    result = factor.applymap(np.sign)
    return result


def get_weight_df_oi_roll_slow(signal_df, factor, ):
    result = (factor.applymap(np.sign)) * (signal_df.applymap(lambda x: 1 if x > 0 else 0))

    return result


# def get_weight_df_no_sign(signal_df, factor_oi):
#     result = signal_df.copy()
#     for i in range(len(signal_df)):
#         for j in range(len(signal_df.columns)):
#             if np.isfinite(signal_df.iloc[i, j]) and not pd.isna(signal_df.iloc[i, j]):
#                 if signal_df.iloc[i, j] > 0:
#                     result.iloc[i, j] = np.sign(factor_oi.iloc[i, j]) * (-1) * signal_df.iloc[i, j]
#                 else:
#                     result.iloc[i, j] = 0
#             else:
#                 result.iloc[i, j] = 0
#     return result
#

def fee_func(df):
    return df.applymap(lambda x: x * ConstFutBasic.transaction_costs_default)


if __name__ == '__main__':
    tickers = ConstFutBasic.fut_code_list
    start_date = '2015-01-01'
    end_date = '2022-09-02'
    threshold_start = 40
    threshold_end = 0

    # price matrix
    roll_df = ExtractDataPostgre.get_delist_date_roll_df(tickers, start_date, end_date,
                                                         non_current_month_commidity_index=True)
    price_df = get_spread_df(tickers, start_date, end_date, roll_df).fillna(method='ffill')
    trade_price_df = get_trade_price_df(tickers, start_date, end_date, roll_df)
    vwap_df = get_vwap_df(tickers, start_date, end_date, roll_df)

    # ttm and ttm_index_df
    ttm = ExtractDataPostgre.get_ttm(tickers=tickers, start_date='2010-01-01', end_date=end_date, as_matrix=False,
                                     roll_index_df=None, non_current_month_commidity_index=True)
    ttm.columns = ['code', 'tradingday', 'TTM']
    # roll month last two day position set 0
    ttm_index_df = ttm.sort_values(['code', 'tradingday'])
    ttm_index_df['index'] = ttm_index_df['TTM'].diff().shift(-1) * ttm_index_df['TTM'].diff().shift(-2)
    ttm_index_df['index'] = ttm_index_df['index'].apply(lambda x: 1 if x == 1 else 0)
    ttm_index_df['TTM'] = ttm_index_df['TTM'] * ttm_index_df['index']
    ttm_index_df = ttm_index_df.pivot_table(index='tradingday', columns='code', values='TTM').applymap(
        lambda x: 1 if threshold_start > x > threshold_end else 0)

    ttm_index_df = format_factor_df(ttm_index_df, price_df)

    # warehouse diff and theta
    wh = ExtractDataPostgre.get_ware_house(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
                                           end_date=end_date, as_matrix=True)
    wh_mu, wh_theta = detrend.detrend_test(wh)
    wh_diff = wh - wh_mu
    wh_diff = format_factor_df(wh_diff, price_df).fillna(0)
    wh_theta = format_factor_df(wh_theta, price_df).fillna(0)

    # cash basis diff and theta
    cb = ExtractDataPostgre.get_cash_basis(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
                                           end_date=end_date, as_matrix=True)
    cb = cb / price_df
    cb_mu, cb_theta = detrend.detrend_test(cb)
    cb_diff = cb - cb_mu
    cb_diff = format_factor_df(cb_diff, price_df).fillna(0)
    cb_theta = format_factor_df(cb_theta, price_df).fillna(0)

    # wh ttm
    # ttm = ExtractDataPostgre.get_ttm(tickers=tickers, start_date='2010-01-01', end_date=end_date, as_matrix=False,
    #                                  roll_index_df=roll_df, non_current_month_commidity_index=True)
    # ttm.columns = ['code', 'tradingday', 'TTM']
    wh = ExtractDataPostgre.get_ware_house(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
                                           end_date=end_date, as_matrix=True)
    wh = wh.fillna(method='ffill', limit=1)
    wh = wh.reset_index().melt(wh.index.name)
    wh.columns = ['tradingday', 'code', 'warehouse']
    wh = wh[['code', 'tradingday', 'warehouse']]
    wh_ttm = FutDataProcess.calculate_ttm_data(data=wh, ttm=ttm, window=None, seasonal=False)
    wh_ttm_seasonal = FutDataProcess.calculate_ttm_data(data=wh, ttm=ttm, window=None, seasonal=True)

    wh_ttm = format_factor_df(wh_ttm.pivot_table(index='tradingday', columns='code', values='TTMvalue'),
                              price_df).fillna(0)
    wh_ttm_seasonal = format_factor_df(
        wh_ttm_seasonal.pivot_table(index='tradingday', columns='code', values='TTMvalue'), price_df).fillna(0)

    # cb ttm
    # ttm = ExtractDataPostgre.get_ttm(tickers=tickers, start_date='2010-01-01', end_date=end_date, as_matrix=False,
    #                                  non_current_month_commidity_index=True)
    # ttm.columns = ['code', 'tradingday', 'TTM']
    cb = ExtractDataPostgre.get_cash_basis(tickers=ConstFutBasic.fut_code_list, start_date='2010-01-01',
                                           end_date=end_date, as_matrix=True)
    cb = cb / price_df
    cb = cb.fillna(method='ffill', limit=1)
    cb = cb.reset_index().melt(cb.index.name)
    cb.columns = ['tradingday', 'code', 'cashbasis']
    cb = cb[['code', 'tradingday', 'cashbasis']]
    cb_ttm = FutDataProcess.calculate_ttm_data(data=cb, ttm=ttm, window=None, seasonal=False)
    cb_ttm_seasonal = FutDataProcess.calculate_ttm_data(data=cb, ttm=ttm, window=None, seasonal=True)
    cb_ttm = format_factor_df(cb_ttm.pivot_table(index='tradingday', columns='code', values='TTMvalue'),
                              price_df).fillna(method='ffill')
    cb_ttm_seasonal = format_factor_df(
        cb_ttm_seasonal.pivot_table(index='tradingday', columns='code', values='TTMvalue'), price_df).fillna(
        method='ffill')

    m1 = 40
    m2 = 10
    window_yoy = 3
    window = 12

    ###########################################
    # factor 1: OI_changed
    file1, file2 = create_factor_oi(window, window_yoy, m1, m2)
    factor_oi_change_rate, factor_oi_change_rate_yoy = get_factor_rate_change_df(file1)
    factor_oi_pct_rate, factor_oi_pct_rate_yoy = get_factor_rate_change_df(file2)

    factor_oi_change_rate = format_factor_df(factor_oi_change_rate, price_df)
    # .applymap(np.sign)
    factor_oi_change_rate_yoy = format_factor_df(factor_oi_change_rate_yoy, price_df)
    # .applymap(np.sign)
    factor_oi_pct_rate = format_factor_df(factor_oi_pct_rate, price_df)
    # .applymap(np.sign)
    factor_oi_pct_rate_yoy = format_factor_df(factor_oi_pct_rate_yoy, price_df)
    # .applymap(np.sign)

    # factor_oi_change_rate_signal = factor_oi_change_rate + factor_oi_change_rate_yoy + factor_oi_pct_rate + factor_oi_pct_rate_yoy

    ##################################
    # factor 2: open interest change and price change
    oi = ExtractDataPostgre.get_fut_code_current_main_and_next_main_ts_data(tickers, start_date, end_date,
                                                                            key_word='oi', as_matrix=True)
    oi = oi.diff()
    # .applymap(np.sign)
    close_price = ExtractDataPostgre.get_fut_code_current_main_and_next_main_ts_data(tickers, start_date, end_date,
                                                                                     key_word='close', as_matrix=True)
    close_price = close_price.diff()
    # .applymap(np.sign)
    factor_oi_signal = close_price * oi
    factor_oi_signal = ExtractDataPostgre.combine_to_con_fut_code_ts(factor_oi_signal, roll_df)

    factor_oi_signal = format_factor_df(factor_oi_signal, price_df)

    ##################################
    # factor save
    factor_list = ['factor_oi_change_rate', 'factor_oi_change_rate_yoy', 'factor_oi_pct_rate', 'factor_oi_pct_rate_yoy',
                   'factor_oi_signal', 'cb_ttm', 'cb_ttm_seasonal', 'wh_ttm', 'wh_ttm_seasonal', 'cb_theta', 'cb_diff',
                   'wh_diff', 'wh_theta']
    factor_save_file_path = 'C:\\Users\\jason.huang\\research\\factor_data\\OI_change\\data\\'
    for factor in factor_list:
        factor_save_file_location = factor_save_file_path + factor + '.csv'
        eval(factor).to_csv(factor_save_file_location)

    #################################

    # single factor test
    factor_list = ['factor_oi_change_rate', 'factor_oi_change_rate_yoy', 'factor_oi_pct_rate', 'factor_oi_pct_rate_yoy',
                   'factor_oi_signal', 'cb_ttm', 'cb_ttm_seasonal', 'wh_ttm', 'wh_ttm_seasonal', 'cb_theta', 'cb_diff',
                   'wh_diff', 'wh_theta']
    for factor in factor_list:
        weight_df = get_weight_df(eval(factor))
        file_location = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\' + factor + '_all.xlsx'
        vol1, vol2 = get_trading_vol(tickers, start_date, end_date)
        trading_schedule_df = get_trading_schedule(vol1, vol2, weight_df.shift(1).fillna(0))
        # 根据trading_schedule 修改weight_df
        weight_df = trading_scheduled_weight(tickers, start_date, end_date,
                                             weight_df.shift(1).fillna(0),
                                             trading_schedule_df)

        roll_index_df = pd.DataFrame(data=0, index=price_df.index, columns=price_df.columns)

        bt1 = BtObj.BacktestStrategy(price_df=price_df, trade_price_df=trade_price_df,
                                     weight_df=weight_df,
                                     previous_main_contract_code_trade_price_df=vwap_df,
                                     current_main_contract_code_trade_price_df=vwap_df,
                                     roll_index_df=roll_index_df,
                                     fee_func=fee_func, shift_index=2,
                                     file_save_location=file_location)
        bt1.write_excel()

        file_location = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\' + factor + '_ttm40.xlsx'
        weight_df = weight_df * ttm_index_df
        bt1 = BtObj.BacktestStrategy(price_df=price_df, trade_price_df=trade_price_df,
                                     weight_df=weight_df,
                                     previous_main_contract_code_trade_price_df=vwap_df,
                                     current_main_contract_code_trade_price_df=vwap_df,
                                     roll_index_df=roll_index_df,
                                     fee_func=fee_func, shift_index=2,
                                     file_save_location=file_location)
        bt1.write_excel()

    # signal graph
    factor_list = ['factor_oi_change_rate', 'factor_oi_change_rate_yoy', 'factor_oi_pct_rate', 'factor_oi_pct_rate_yoy',
                   'factor_oi_signal', 'cb_ttm', 'cb_ttm_seasonal', 'wh_ttm', 'wh_ttm_seasonal', 'cb_theta', 'cb_diff',
                   'wh_diff', 'wh_theta']
    for factor in factor_list:

        graph_file_path = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\all_factor\\signal_graph\\'
        try:
            f = eval(factor) * (get_weight_df(eval(factor)).applymap(np.abs))
        except Exception as e:
            f = LoadLocalData.load_local_factor_csv(
                file_location='C:\\Users\\jason.huang\\research\\factor_data\\OI_change\\data\\' + factor + '.csv')
            f = f * (get_weight_df(f).applymap(np.abs))

        FutDataProcess.single_factor_signal_graph_group(signal=f, file_path=graph_file_path, factor_name=factor,
                                                        group_list=None, pnl=None,
                                                        price_df=price_df, lag=1)

        try:
            f2 = eval(factor) * (get_weight_df(eval(factor)).applymap(np.abs)) * ttm_index_df
        except Exception as e:
            f2 = LoadLocalData.load_local_factor_csv(
                file_location='C:\\Users\\jason.huang\\research\\factor_data\\OI_change\\data\\' + factor + '.csv')
            f2 = f2 * (get_weight_df(f).applymap(np.abs)) * ttm_index_df
        # f2 = eval(factor) * (get_weight_df(eval(factor)).applymap(np.abs)) * ttm_index_df
        FutDataProcess.single_factor_signal_graph_group(signal=f, file_path=graph_file_path,
                                                        factor_name=factor + '_ttm40', group_list=None, pnl=None,
                                                        price_df=price_df, lag=1)

    factor_list = ['factor_oi_signal', 'cb_ttm', 'cb_ttm_seasonal', 'wh_ttm', 'wh_ttm_seasonal', 'cb_theta', 'cb_diff',
                   'wh_diff', 'wh_theta']
    factor_oi_change_rate_signal = factor_oi_change_rate.applymap(np.sign) + factor_oi_change_rate_yoy.applymap(
        np.sign) + factor_oi_pct_rate.applymap(np.sign) + factor_oi_pct_rate_yoy.applymap(np.sign)
    for factor in factor_list:
        weight_df = get_weight_df_oi_roll_slow(factor_oi_change_rate_signal, eval(factor))
        file_location = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\OI_roll_slow_case\\' + factor + '_all.xlsx'
        vol1, vol2 = get_trading_vol(tickers, start_date, end_date)
        trading_schedule_df = get_trading_schedule(vol1, vol2, weight_df.shift(1).fillna(0))
        # 根据trading_schedule 修改weight_df
        weight_df = trading_scheduled_weight(tickers, start_date, end_date,
                                             weight_df.shift(1).fillna(0),
                                             trading_schedule_df)

        roll_index_df = pd.DataFrame(data=0, index=price_df.index, columns=price_df.columns)

        bt1 = BtObj.BacktestStrategy(price_df=price_df, trade_price_df=trade_price_df,
                                     weight_df=weight_df,
                                     previous_main_contract_code_trade_price_df=vwap_df,
                                     current_main_contract_code_trade_price_df=vwap_df,
                                     roll_index_df=roll_index_df,
                                     fee_func=fee_func, shift_index=2,
                                     file_save_location=file_location)
        bt1.write_excel()

        file_location = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\OI_roll_slow_case\\' + factor + '_ttm40.xlsx'
        weight_df = weight_df * ttm_index_df
        bt1 = BtObj.BacktestStrategy(price_df=price_df, trade_price_df=trade_price_df,
                                     weight_df=weight_df,
                                     previous_main_contract_code_trade_price_df=vwap_df,
                                     current_main_contract_code_trade_price_df=vwap_df,
                                     roll_index_df=roll_index_df,
                                     fee_func=fee_func, shift_index=2,
                                     file_save_location=file_location)
        bt1.write_excel()

    # signal graph
    factor_list = ['factor_oi_signal', 'cb_ttm', 'cb_ttm_seasonal', 'wh_ttm', 'wh_ttm_seasonal', 'cb_theta', 'cb_diff',
                   'wh_diff', 'wh_theta']
    factor_oi_change_rate_signal = factor_oi_change_rate.applymap(np.sign) + factor_oi_change_rate_yoy.applymap(
        np.sign) + factor_oi_pct_rate.applymap(np.sign) + factor_oi_pct_rate_yoy.applymap(np.sign)
    for factor in factor_list:
        graph_file_path = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\OI_roll_slow_case\\signal_graph\\'
        try:
            f = eval(factor) * (get_weight_df(eval(factor)).applymap(np.abs))
        except Exception as e:
            f = LoadLocalData.load_local_factor_csv(
                file_location='C:\\Users\\jason.huang\\research\\factor_data\\OI_change\\data\\' + factor + '.csv')
            f = f * (get_weight_df(f).applymap(np.abs))

        FutDataProcess.single_factor_signal_graph_group(signal=f, file_path=graph_file_path, factor_name=factor,
                                                        group_list=None, pnl=None,
                                                        price_df=price_df, lag=1)
        try:
            f2 = eval(factor) * (get_weight_df(eval(factor)).applymap(np.abs)) * ttm_index_df
        except Exception as e:
            f2 = LoadLocalData.load_local_factor_csv(
                file_location='C:\\Users\\jason.huang\\research\\factor_data\\OI_change\\data\\' + factor + '.csv')
            f2 = f2 * (get_weight_df(f).applymap(np.abs)) * ttm_index_df
        # f2 = eval(factor) * (get_weight_df(eval(factor)).applymap(np.abs)) * ttm_index_df
        FutDataProcess.single_factor_signal_graph_group(signal=f2, file_path=graph_file_path,
                                                        factor_name=factor + '_ttm40', group_list=None, pnl=None,
                                                        price_df=price_df, lag=1)

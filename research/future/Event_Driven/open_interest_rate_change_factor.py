import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import model.factor.technical_indicator as TechIndicator
from my_stats.time_series import time_series_regression as TimeSeriesReg
import data.Process.fut_data_process as DataProcessFut
import data.ConstantData.future_basic_information as ConstFutBasic
import data.SQL.extract_data_from_postgre as ExtractDataPostgre


def get_return_df(tickers, start_date, end_date, key_word):
    pair_df, df = ExtractDataPostgre.get_term_structure_series(tickers=tickers, start_date=start_date,
                                                               end_date=end_date, key_word=key_word)
    pair_spread, price_cost_df = ExtractDataPostgre.get_pair_series(df=df, pair_df_all=pair_df, method='spread')
    # pair_spread = pair_spread.diff()
    price_df = ExtractDataPostgre.joint_con_ts(pair_df_all=pair_df, pair_spread=pair_spread)
    price_cost_df = ExtractDataPostgre.joint_con_ts(pair_df_all=pair_df, pair_spread=price_cost_df)
    return price_df, price_cost_df


def power_law_index(array):
    s = [np.log(x) for x in array / np.min(array)]
    return 1 / np.sum(s[:5]) * len(array[:5]) + 1


def extract_open_interest_data(tickers, start_date, end_date):
    tickers_list = ExtractDataPostgre.get_code_instrument_mapping()
    tickers_list = [key for key, value in tickers_list.items() if value == tickers]
    open_interest_matrix = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list, start_date=start_date,
                                                                       end_date=end_date, key_word='oi', as_matrix=True)
    open_interest_daily = open_interest_matrix.apply(np.nansum, axis=1).reset_index()
    open_interest_matrix = open_interest_matrix.reset_index()
    n_col = len(open_interest_matrix.columns)
    for i in range(len(open_interest_matrix)):
        seq_list = np.sort(np.array(open_interest_matrix.iloc[i, 1:n_col].replace(0, np.nan).dropna()))
        open_interest_daily.loc[i, 'OI_1'] = seq_list[-1] / open_interest_daily.iloc[i, 1]
        open_interest_daily.loc[i, 'OI_2'] = seq_list[-2] / open_interest_daily.iloc[i, 1]
        open_interest_daily.loc[i, 'OI_3'] = seq_list[-3] / open_interest_daily.iloc[i, 1]
        open_interest_daily.loc[i, 'OI_4'] = seq_list[-4] / open_interest_daily.iloc[i, 1]
        open_interest_daily.loc[i, 'OI_5'] = seq_list[-5] / open_interest_daily.iloc[i, 1]
        open_interest_daily.loc[i, 'OI_power'] = power_law_index(seq_list)

    price_df, _ = get_return_df(tickers, start_date, end_date, key_word='close')

    roll_index = ExtractDataPostgre.roll_index(tickers=tickers, start_date=start_date, end_date=end_date).reset_index()
    data = pd.merge(left=open_interest_daily, right=price_df.reset_index(), how='left', left_on='trade_date',
                    right_on='tradingday')
    data = pd.merge(left=data, right=roll_index, how='left', on='tradingday')

    pair_df, df = ExtractDataPostgre.get_term_structure_series(tickers=tickers, start_date=start_date,
                                                               end_date=end_date, key_word='oi')
    main_contract_list = np.unique(pair_df['current_main_instrumentid'])


def graph_plot_of_one_future(df_oi1, df_oi2, df_close, df_spread, tickers, file_location):
    df_oi1.index = range(len(df_oi1))
    df_oi1.columns.name = None
    df_oi2.index = range(len(df_oi2))
    df_close.index = range(len(df_close))
    df_spread.index = range(len(df_spread))
    fig, axs = plt.subplots(4, figsize=(40, 30))
    axs[0].plot(df_oi1.iloc[:, 0].values, df_oi1.iloc[:, 1:].values, label=df_oi1.columns[1:])
    axs[0].set_title(tickers + 'oi')
    axs[1].plot(df_oi2.iloc[:, 0].values, df_oi2.iloc[:, 1:].values, label=df_oi2.columns[1:])
    axs[1].set_title(tickers + 'oi_pct')
    axs[2].plot(df_close.iloc[:, 0].values, df_close.iloc[:, 1:].values, label=df_close.columns[1:])
    axs[2].set_title(tickers + 'close_price')
    axs[3].plot(df_spread.iloc[:, 0].values, df_spread.iloc[:, 1:].values, label=df_spread.columns[1:])
    axs[3].set_title(tickers + 'spread')

    plt.savefig(file_location, bbox_inches='tight', dpi=200)
    plt.close()


def extract_open_interest_data_based_on_contract(tickers_list_all, start_date1, end_date):
    index_length_df = pd.DataFrame()
    result_all = pd.DataFrame()
    delist_date_df = ExtractDataPostgre.get_list_date_and_delist_date_of_instrument()

    for a in range(len(tickers_list_all)):
        try:
            tickers = tickers_list_all[a]
            trading_calendar = ExtractDataPostgre.get_trading_calendar(exchange=tickers.split('.')[-1])
            start_date = DataProcessFut.adjust_start_date(start_date1, tickers, threshold=1, delta_time=365)

            pair_df, df_oi = ExtractDataPostgre.get_term_structure_series(tickers=tickers, start_date=start_date,
                                                                          end_date=end_date, key_word='oi')
            pair_spread, df_close = ExtractDataPostgre.get_term_structure_series(tickers=tickers, start_date=start_date,
                                                                                 end_date=end_date, key_word='close')
            df_spread, _ = get_return_df(tickers, start_date, end_date, key_word='close')
            df_spread = df_spread.reset_index()
            df_oi1 = df_oi.copy().reset_index()
            df_close = df_close.reset_index()
            for j in range(len(df_close) - 4):
                try:
                    index_end = np.max(np.where(df_close.iloc[:, j + 1] > 0))
                    df_close.iloc[:index_end, j + 3] = np.nan
                except Exception as e:
                    pass

            tickers_list = ExtractDataPostgre.get_code_instrument_mapping()
            tickers_list = [key for key, value in tickers_list.items() if value == tickers]
            open_interest_matrix = ExtractDataPostgre.get_future_daily_data_sm(tickers=tickers_list,
                                                                               start_date=start_date,
                                                                               end_date=end_date, key_word='oi',
                                                                               as_matrix=True)
            open_interest_daily = open_interest_matrix.apply(np.nansum, axis=1).reset_index()
            open_interest_daily.columns = ['trade_date', 'oi_all']
            df_oi = df_oi.reset_index()
            df_oi = pd.merge(left=df_oi, right=open_interest_daily, how='left', on='trade_date')
            for i in range(1, len(df_oi.columns) - 1):
                df_oi.loc[:, df_oi.columns[i] + '_rate'] = df_oi.iloc[:, i] / df_oi['oi_all']
            pct_columns = [i for i in df_oi.columns if i[-4:] == 'rate']
            df_oi2 = df_oi[['trade_date'] + pct_columns]

            graph_file_location = "C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\graph\\" + tickers + ".png"
            graph_plot_of_one_future(df_oi1, df_oi2, df_close, df_spread, tickers, file_location=graph_file_location)

            max_pct_index = pd.DataFrame()
            for j in range(len(pct_columns)):
                max_pct_index.loc[j, 'code'] = pct_columns[j][:-5]
                max_pct_index.loc[j, 'max_index'] = \
                    df_oi[pct_columns[j]][df_oi[pct_columns[j]] == np.nanmax(df_oi[pct_columns[j]])].index[0]

            time_length = 2 * round(max_pct_index['max_index'].diff().mean())
            index_length_df.loc[a, 'code'] = tickers
            index_length_df.loc[a, 'time_length'] = time_length

            data = pd.DataFrame(index=range(time_length + 1),
                                columns=[i[:-5] for i in df_oi.columns if i[-4:] == 'rate'])

            for j in range(len(data.columns)):
                code = data.columns[j]
                delist_date = delist_date_df[delist_date_df['ts_code'] == code]['delist_date'].values[0]
                if delist_date in list(df_oi['trade_date']):
                    index_date = np.where(df_oi['trade_date'] == delist_date)[0][0]
                    if index_date > time_length:
                        data.loc[:, code] = np.array(df_oi[code + '_rate'][(index_date - time_length):(index_date + 1)])
                    else:
                        data.loc[:, code] = np.array(
                            [np.nan] * (time_length - index_date) + list(df_oi[code + '_rate'][:(index_date + 1)]))
                else:
                    try:
                        index_delist_date = np.where(trading_calendar['cal_date'] == delist_date)[0][0]
                        index_last_date = \
                            np.where(trading_calendar['cal_date'] == df_oi['trade_date'][len(df_oi) - 1])[0][
                                0]
                        delta_days = index_delist_date - index_last_date
                        data.loc[:, code] = np.array(
                            list(df_oi[code + '_rate'][-1 * (time_length - delta_days + 1):]) + [np.nan] * delta_days)
                    except Exception as e:
                        pass
            data_oi_rate = pd.DataFrame(index=range(time_length + 1),
                                        columns=[i[:-5] for i in df_oi.columns if i[-4:] == 'rate'])

            for j in range(len(data_oi_rate.columns)):
                code = data_oi_rate.columns[j]
                delist_date = delist_date_df[delist_date_df['ts_code'] == code]['delist_date'].values[0]
                if delist_date in list(df_oi['trade_date']):
                    index_date = np.where(df_oi['trade_date'] == delist_date)[0][0]
                    if index_date > time_length:
                        data_oi_rate.loc[:, code] = np.array(df_oi[code][(index_date - time_length):(index_date + 1)])
                    else:
                        data_oi_rate.loc[:, code] = np.array(
                            [np.nan] * (time_length - index_date) + list(df_oi[code][:(index_date + 1)]))
                else:
                    try:
                        index_delist_date = np.where(trading_calendar['cal_date'] == delist_date)[0][0]
                        index_last_date = \
                            np.where(trading_calendar['cal_date'] == df_oi['trade_date'][len(df_oi) - 1])[0][
                                0]
                        delta_days = index_delist_date - index_last_date
                        data_oi_rate.loc[:, code] = np.array(
                            list(df_oi[code][-1 * (time_length - delta_days + 1):]) + [np.nan] * delta_days)
                    except Exception as e:
                        pass
            for j in range(len(data_oi_rate.columns)):
                data_oi_rate.iloc[:, j] = np.array(data_oi_rate.iloc[:, j]) / np.array(
                    [time_length + 1 - 10 - i if time_length + 1 - 10 - i > 0 else np.nan for i in
                     range(time_length + 1)])

            data_time = pd.DataFrame(index=range(time_length + 1),
                                     columns=[i[:-5] for i in df_oi.columns if i[-4:] == 'rate'])
            for j in range(len(data_time.columns)):
                try:
                    code = data_time.columns[j]
                    delist_date = delist_date_df[delist_date_df['ts_code'] == code]['delist_date'].values[0]
                    index_delist_date = np.where(trading_calendar['cal_date'] == delist_date)[0][0]
                    data_time.loc[:, code] = np.array(
                        trading_calendar['cal_date'][(index_delist_date - time_length):(index_delist_date + 1)])
                except Exception as e:
                    pass
            data_time.to_csv(
                "C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\date_time\\" + tickers + ".csv")
            data_oi_rate.to_csv(
                "C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_rate\\" + tickers + ".csv")

            data.to_csv(
                "C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_data_matrix\\" + tickers + ".csv")
            result = pd.DataFrame()
            for j in range(len(data.columns)):
                try:
                    array = pd.DataFrame(data.iloc[:, j]).dropna()
                    params, pvalues, rsquared = TimeSeriesReg.order_regression_with_time(array=array.values, order=2)
                    result.loc[j, 'code'] = data.columns[j]
                    result.loc[j, 'r2'] = rsquared
                    result.loc[j, 'const'] = params['const']
                    result.loc[j, 't1'] = params['t1']
                    result.loc[j, 't2'] = params['t2']
                    result.loc[j, 'const_p_value'] = pvalues['const']
                    result.loc[j, 't1_p_value'] = pvalues['t1']
                    result.loc[j, 't2_p_value'] = pvalues['t2']
                except Exception as e:
                    pass
            result_all = result_all.append(result)
        except Exception as e:
            pass
    result_all.to_csv(
        "C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\regression_all.csv")
    index_length_df.to_csv(
        "C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\time_length.csv")


def calculate_open_interest_diff():
    window = 5
    window_yoy=3
    file_path = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_rate\\'
    file_path1 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_diff\\OI_rate_rolling_window'+str(window)+'\\'
    file_path2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\OI_diff\\OI_rate_rolling_month'+str(window_yoy)+'\\'
    if not os.path.exists(file_path1):
        os.makedirs(file_path1)
    if not os.path.exists(file_path2):
        os.makedirs(file_path2)
    for file_name in os.listdir(file_path):
        file_location = file_path + file_name
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
                    result2.iloc[i, j] = data.iloc[i, j] - np.nanmean(data.iloc[i, [j - 3, j - 6, j - 9]])
                except Exception as e:
                    pass
        file_location1 = file_path1 + file_name
        file_location2 = file_path2 + file_name
        result1.to_csv(file_location1)
        result2.to_csv(file_location2)


def get_diff_ts():
    file_path_time = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\date_time\\'
    file_path_r1 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\\OI_diff\\OI_rate_rolling_window\\'
    file_path_r2 = 'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\open_interest_signa\\\OI_diff\\OI_rate_rolling_month\\'
    result_all = pd.DataFrame()
    for file_name in os.listdir(file_path_time):
        file_location_time = file_path_time + file_name
        file_location_r1 = file_path_r1 + file_name
        file_location_r2 = file_path_r2 + file_name
        dt = pd.read_csv(file_location_time, index_col=0)
        r1 = pd.read_csv(file_location_r1, index_col=0)
        r2 = pd.read_csv(file_location_r2, index_col=0)
        result = pd.DataFrame()
        k = 0
        for i in range(round(max(len(dt) * 0.6, len(dt) - 30)), len(dt) - 15):
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
        'C:\\Users\\jason.huang\\research\\desciptive_statistics\\term_structure\\event_driven\\signal_oi_rate.csv')


if __name__ == '__main__':
    tickers = 'RB.SHF'
    tickers_list_all = ConstFutBasic.fut_code_list
    start_date1 = '2010-02-01'
    end_date = '2022-08-22'

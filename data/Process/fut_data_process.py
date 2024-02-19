import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data.ConstantData.future_basic_information as ConstFutBasic
import data.SQL.extract_data_from_postgre as ExtractDataPostgre


# 分钟级数据删除开盘和收盘前几分钟
def delete_open_close_data(df, tickers, open_filter=5, close_filter=5):
    df['hour_minute'] = df['Trade_DT'].apply(lambda x: x.hour * 100 + x.minute)
    time_list = ConstFutBasic.fut_code_trading_min_time_dict[tickers]
    filter_time = []
    for i in range(len(time_list)):
        open_time = time_list[i][0]
        close_time = time_list[i][1]
        filter_time = filter_time + [int(open_time[:2]) * 100 + int(open_time[-2:]) + k for k in range(open_filter)]
        if close_time[-2:] == '00':
            if close_time[:2] == '00':
                filter_time = filter_time + [int(close_time[:2]) * 100 + 2400 + int(close_time[-2:]) - k - 40 for k in
                                             range(1, close_filter)] + [
                                  int(close_time[:2]) * 100 + int(close_time[-2:])]
            else:
                filter_time = filter_time + [int(close_time[:2]) * 100 + int(close_time[-2:]) - k - 40 for k in
                                             range(1, close_filter)] + [
                                  int(close_time[:2]) * 100 + int(close_time[-2:])]
        else:
            filter_time = filter_time + [int(close_time[:2]) * 100 + int(close_time[-2:]) - k for k in
                                         range(close_filter)]

    df = df[~df['hour_minute'].isin(filter_time)]
    df = df[df.columns[:-1]]
    df.index = range(len(df))
    return df


# 上市未满一年的品种从上市满一年后截断数据
def adjust_start_date(start_date, tickers, threshold=1, delta_time=365):
    list_date = ConstFutBasic.fut_code_listing_date[tickers]
    list_date_datetime = datetime.datetime.strptime(list_date, '%Y-%m-%d')
    delta_days = list_date_datetime - datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if delta_days.days + delta_time > 0:
        list_date_datetime = list_date_datetime.replace(year=list_date_datetime.year + threshold)
        start_date = datetime.datetime.strftime(list_date_datetime, "%Y-%m-%d")
    return start_date


# 删除上市未满一年的data
def delete_new_product_data(df_matrix, start_date):
    df = df_matrix.copy()
    for i in range(df.shape[1]):
        code = df.columns[i]
        new_start_date = adjust_start_date(start_date=start_date, tickers=code)
        if new_start_date > start_date:
            new_start_date = datetime.date(int(new_start_date.split('-')[0]), int(new_start_date.split('-')[1]),
                                           int(new_start_date.split('-')[2]))
            df.loc[df.index < new_start_date, code] = np.nan
    return df


# 因子矩阵标准化对齐
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


# 获取下一个主力合约
def get_next_main_contract(tickers):
    fut_code = ExtractDataPostgre.get_code_instrument_mapping()[tickers]
    roll_dict = ConstFutBasic.fut_code_roll_instrument[fut_code]
    month = tickers.split('.')[0][-2:]
    year = tickers.split('.')[0][-4:-2]

    new_month = roll_dict[month]
    if int(new_month) < int(month):
        year = str(int(year) + 1)
    new_tickers = tickers.split('.')[0][:-4] + year + new_month + '.' + ''.join(tickers.split('.')[1:])
    return new_tickers


# 交易间隔时长（天数）
def calculate_trading_timedelta_of_two_dates(date1, date2):
    if type(date1) is str:
        pass
    else:
        try:
            date1 = datetime.datetime.strftime(date1, '%Y-%m-%d')
        except Exception as e:
            raise e
    if type(date2) is str:
        pass
    else:
        try:
            date2 = datetime.datetime.strftime(date2, '%Y-%m-%d')
        except Exception as e:
            raise e

    calendar_df = ExtractDataPostgre.get_trading_calendar()
    calendar_df = np.array(
        [datetime.datetime.strftime(calendar_df.iloc[i, 0], "%Y-%m-%d") for i in range(len(calendar_df))])
    try:
        index1 = np.where(calendar_df == date1)[0]
        if len(index1) == 0:
            if date1 > calendar_df[-1]:
                index1 = None
            else:
                index1 = np.where(calendar_df <= date1)[0]
                index1 = index1[-1]
        else:
            index1 = index1[-1]
    except Exception as e:
        index1 = None
    try:
        index2 = np.where(calendar_df == date2)[0]
        if len(index2) == 0:
            if date2 > calendar_df[-1]:
                index2 = None
            else:
                index2 = np.where(calendar_df <= date2)[0]
                index2 = index2[-1]
        else:
            index2 = index2[-1]
    except Exception as e:
        index2 = None

    if index1 is None or index2 is None:
        time_delta = datetime.datetime.strptime(date2, '%Y-%m-%d') - datetime.datetime.strptime(date1, '%Y-%m-%d')
        time_delta = round(time_delta.days * 5 / 7)
    else:
        time_delta = index2 - index1
    return time_delta


# 计算TTM YOY数据
def calculate_ttm_data(data, ttm, window=None, seasonal=False):
    df = pd.merge(left=data, right=ttm, how='left', left_on=[data.columns[0], data.columns[1]],
                  right_on=[ttm.columns[0], ttm.columns[1]])
    df.columns = ['code', 'tradingday'] + list(df.columns[2:])
    tickers_list = np.unique(df['code'])
    result = pd.DataFrame()
    for tickers in tickers_list:
        if window is None:
            if tickers in ConstFutBasic.group_monthly_roll + ConstFutBasic.group_cffex_stock:
                if seasonal:
                    window = ConstFutBasic.window_yoy_monthly_roll
                else:
                    window = ConstFutBasic.window_monthly_roll
            else:
                if seasonal:
                    window = ConstFutBasic.window_yoy_non_monthly_roll
                else:
                    window = ConstFutBasic.window_non_monthly_roll
        df_tickers = df[df['code'] == tickers].sort_values('tradingday')
        df_tickers.index = range(len(df_tickers))
        for i in range(len(df_tickers)):
            ttm_value = df_tickers.loc[i, 'TTM']
            trade_date = df_tickers.loc[i, 'tradingday']
            df_tickers_select = df_tickers[df_tickers['TTM'] == ttm_value]
            df_tickers_select = df_tickers_select[df_tickers_select['tradingday'] < trade_date]
            if seasonal:
                df_tickers_select['month'] = df_tickers_select['tradingday'].apply(lambda x: x.month)
                df_tickers_select = df_tickers_select[df_tickers_select['month'] == trade_date.month]
            if len(df_tickers_select) < window:
                df_tickers.loc[i, 'TTMvalue'] = np.nan
            else:
                try:
                    mean_value = np.nanmean(np.array(df_tickers_select[df_tickers_select.columns[2]])[-1 * window:])
                    df_tickers.loc[i, 'TTMvalue'] = df_tickers.loc[i, df_tickers.columns[2]] - mean_value
                except Exception as e:
                    df_tickers.loc[i, 'TTMvalue'] = np.nan
        result = pd.concat([result, df_tickers])
        print(tickers + ' is ok')
    result = result[['code', 'tradingday', 'TTMvalue']]
    return result


# 测试单因子信号图

def single_factor_signal_graph(signal, file_location, pnl=None, price_df=None, lag=1):
    if pnl is None and price_df is None:
        return
    if price_df is not None:
        pnl = price_df.shift(-1 * lag) - price_df

    n = 6
    fig, axs = plt.subplots((len(signal.columns) // n + 1) * 2, n,
                            figsize=(15 * n, 15 * (len(signal.columns) // n + 1) * 2))

    for i in range(len(signal.columns)):
        code = signal.columns[i]
        try:

            data = pd.DataFrame()
            data['signal'] = signal.iloc[:, i]
            data['pnl'] = pnl.iloc[:, i]
            data = data.dropna()
            data.index = range(len(data))
            if len(data) > n:
                data = data.sort_values('signal')
                data['cum_pnl'] = data['pnl'].cumsum()
                str1 = code + ' cumpnl'
                str2 = code + ' signal v.s. cumpnl'
                axs[(i // n) * 2, i % n].plot(range(len(data)), data['cum_pnl'], color='r')
                axs[(i // n) * 2, i % n].set_title(str1)
                axs[(i // n) * 2 + 1, i % n].plot(data['signal'], data['cum_pnl'])
                axs[(i // n) * 2 + 1, i % n].set_title(str2)

            else:
                pass
            print(code + ' is ok')
        except Exception as e:
            print(code + ' is not ok')

    plt.savefig(file_location, bbox_inches='tight')
    plt.close()


def single_factor_signal_graph_group(signal, file_path, factor_name, group_list=None, pnl=None, price_df=None, lag=1,
                                     drop_zero=True):
    if pnl is None and price_df is None:
        return
    if price_df is not None:
        pnl = price_df.shift(-1 * lag) - price_df
    if group_list is None:
        group_list = ['cm_cn_group_grains_oilseeds', 'cm_cn_group_livestock', 'cm_cn_group_softs',
                      'cm_cn_group_base_metal', 'cm_cn_group_black', 'cm_cn_group_chemicals', 'cm_cn_group_energy',
                      'cm_cn_group_stock_index', 'cm_cn_group_interest_rate']
    for group in group_list:
        group_vals = [ConstFutBasic.fut_simple_code_to_windcode_mapping_dict[x] for x in eval('ConstFutBasic.' + group)]
        save_location = file_path + factor_name + '_' + group + '.png'
        # if len(group_vals) <= 3:
        #     n = 1
        # else:
        #     n = 3
        n = 3
        fig, axs = plt.subplots((len(group_vals) // n + 1) * 2, n,
                                figsize=(15 * n, 15 * (len(group_vals) // n + 1) * 2))
        for i in range(len(group_vals)):
            code = group_vals[i]
            try:

                data = pd.DataFrame()
                data['signal'] = signal.loc[:, group_vals[i]]
                data['pnl'] = pnl.loc[:, group_vals[i]]
                if drop_zero:
                    data['signal'] = data['signal'].replace(0, np.nan)
                data = data.dropna()
                data.index = range(len(data))
                if len(data) > n:
                    data = data.sort_values('signal')
                    data['cum_pnl'] = data['pnl'].cumsum()
                    str1 = code + ' cumpnl'
                    str2 = code + ' signal v.s. cumpnl'
                    axs[(i // n) * 2, i % n].plot(range(len(data)), data['cum_pnl'], color='r')
                    axs[(i // n) * 2, i % n].set_title(str1)
                    axs[(i // n) * 2 + 1, i % n].plot(data['signal'], data['cum_pnl'])
                    axs[(i // n) * 2 + 1, i % n].axvline(0, color='r')
                    axs[(i // n) * 2 + 1, i % n].set_title(str2)

                else:
                    pass
                print(code + ' is ok')
            except Exception as e:
                print(code + ' is not ok')

        plt.savefig(save_location, bbox_inches='tight')
        plt.close()






import time

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import datetime

import data.ConstantData.future_basic_information as ConstFutBasic
import data.Process.fut_data_process as FutDataProcess


######################################################
#
# basic function
#
######################################################
# 从security master数据库中提取数据
def sql_query_from_sm(sql_statement: str):
    """
    :param sql_statement: string
    :return: a dataframe
    """
    dbconn = create_engine("postgresql+psycopg2://securitymaster:sm_ppt_2021@192.168.208.20:5432/securitymaster",
                           echo=False)
    try:
        tmp_pd = pd.read_sql_query(sql_statement, con=dbconn)
    except Exception as e:
        tmp_pd = pd.DataFrame()
    return tmp_pd


# 从quant analysis数据库中提取数据
def sql_query_from_qa(sql_statement: str):
    """
    :param sql_statement: string
    :return: a dataframe
    """
    dbconn = create_engine("postgresql+psycopg2://quantanalysis:qa_ppt_2021@192.168.208.20:5432/quantanalysis",
                           echo=False)
    try:
        tmp_pd = pd.read_sql_query(sql_statement, con=dbconn)
    except Exception as e:
        tmp_pd = pd.DataFrame()
    return tmp_pd


# 从Postgre数据库提取数据
def sql_query(sql_statement: str, db_engine: str):
    """
    sql query from different database
    :param sql_statement: sql query statement
    :param db_engine: quant analysis database or security master database
    :return: dataframe result of sql query
    """
    if db_engine == 'quantanalysis' or db_engine == 'qa':
        return sql_query_from_qa(sql_statement)
    elif db_engine == 'securitymaster' or db_engine == 'sm':
        return sql_query_from_sm(sql_statement)
    else:
        raise NameError('db_enginee can not be visted')


######################################################

######################################################
#
# time series basic function
#
######################################################

# 提取时间序列数据
def get_ts_data(tickers, start_date, end_date, key_word, table_name, db_engine, code_str, trade_dt_str, code_str_filter,
                trade_dt_filter_str, other_condition=None, as_matrix=False):
    if type(tickers) is str:
        tickers_str = tickers
    elif type(tickers) is list:
        tickers_str = '\',\''.join(tickers)
    else:
        raise TypeError('Input of tickers should be str or list')

    if type(key_word) is str:
        key_word_str = key_word
    elif type(key_word) is list:
        key_word_str = ','.join(key_word)
    else:
        raise TypeError('Input of key_word should be str or list')

    sql_statement = f"SELECT {code_str},{trade_dt_str},{key_word_str} from {table_name} where {code_str_filter} in (\'{tickers_str}\') and {trade_dt_filter_str}>=\'{start_date}\' and {trade_dt_filter_str}<=\'{end_date}\' "
    if other_condition is not None:
        sql_statement = sql_statement + ' and ' + other_condition

    data = sql_query(sql_statement, db_engine=db_engine)
    data = data.sort_values([trade_dt_str])
    data.index = range(len(data))
    if as_matrix:
        data = data.pivot_table(values=key_word, index=trade_dt_str, columns=code_str)
    return data


# 从security master库中提取日频时间序列
def get_future_daily_data_sm(tickers, start_date, end_date, key_word, table_name="future.t_marketdata_daily",
                             db_engine='securitymaster', code_str='ts_code', trade_dt_str='trade_date',
                             code_str_filter='ts_code', trade_dt_filter_str='trade_date', other_condition=None,
                             as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str,
                       other_condition=other_condition, as_matrix=as_matrix)


# 从security master库中提取分钟频率时间序列
def get_future_minute_data_sm(tickers, start_date, end_date, key_word, table_name="future.t_marketdata_minute",
                              db_engine='securitymaster', code_str='windcode', trade_dt_str='datetime',
                              code_str_filter='windcode', trade_dt_filter_str='tradingday', other_condition=None,
                              as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str,
                       other_condition=other_condition, as_matrix=as_matrix)


# 从security master库中提取分钟频率时间序列
def get_future_minute_aggregation_data_sm(tickers, start_date, end_date, key_word,
                                          table_name="future.t_marketdata_minute_aggregations",
                                          db_engine='securitymaster', code_str='windcode', trade_dt_str='datetime',
                                          code_str_filter='windcode', trade_dt_filter_str='tradingday',
                                          other_condition=None,
                                          as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str,
                       other_condition=other_condition, as_matrix=as_matrix)


# 从quant analysis库中提取日品时间序列（主力合约）
def get_future_daily_data_qa(tickers, start_date, end_date, key_word, table_name="future.t_cm_gsci_marketdata_daily_ts",
                             db_engine='quantanalysis', code_str='instrumentid', trade_dt_str='tradingday',
                             code_str_filter='main_contract_code',
                             trade_dt_filter_str='tradingday', other_condition=None, as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str,
                       other_condition=other_condition, as_matrix=as_matrix)


# 获取每天的主力合约或次主力合约
def get_daily_contract(tickers, start_date, end_date, key_word=None, index=1, roll_method='gsci'):
    """
    a function of get the current main instrument / next main instrument/previous main instrument of the fut_code
    :param roll_method: roll method: gsci, oi, trade_vol
    :param tickers: a list of fut_code / one fut_code
            type: list, str
            e.g. : ['A.DCE','AL.SHF',...] / 'A.DCE'
    :param start_date: string, format: %Y-%m-%d
            e.g. :  '2020-01-01'
    :param end_date:  string, format: %Y-%m-%d
            e.g. :  '2020-01-01'
    :param key_word: default None
    :param index: int
            0 : previous main instrument
            1 : current main instrument
            2 : next main instrument
    :return:  a dataframe
            e.g. :        main_contract_code    tradingday         current_main_instrumentid
                0                A.DCE          2020-01-02                 A2005.DCE
                1                A.DCE          2020-01-03                 A2005.DCE
                2                A.DCE          2020-01-06                 A2005.DCE
                3                A.DCE          2020-01-07                 A2005.DCE

    """
    if roll_method == 'gsci':
        if index == 1:
            key_word = 'current_main_instrumentid'
        elif index == 2:
            key_word = 'next_main_instrumentid'
        elif index == 0:
            key_word = 'previous_main_instrumentid'
        else:
            pass
        df = get_ts_data(tickers=tickers, start_date=start_date, end_date=end_date, key_word=key_word,
                         table_name='future.t_cm_gsci_main_contract_map_daily', db_engine='quantanalysis',
                         code_str='main_contract_code', trade_dt_str='tradingday', code_str_filter='main_contract_code',
                         trade_dt_filter_str='tradingday', other_condition=None, as_matrix=False)
        df = df.dropna().reset_index(drop=True)
    elif roll_method == 'oi':
        if index == 1:
            key_word = 'current_main_instrumentid'
        elif index == 2:
            key_word = 'next_main_instrumentid'
        elif index == 0:
            key_word = 'previous_main_instrumentid'
        else:
            pass
        df = get_ts_data(tickers=tickers, start_date=start_date, end_date=end_date, key_word=key_word,
                         table_name='future.t_oi_main_contract_map_daily', db_engine='quantanalysis',
                         code_str='main_contract_code', trade_dt_str='tradingday', code_str_filter='main_contract_code',
                         trade_dt_filter_str='tradingday', other_condition=None, as_matrix=False)
        df = df.dropna().reset_index(drop=True)
    elif roll_method == 'trade_vol':
        if index == 1:
            key_word = 'current_main_instrumentid'
        elif index == 2:
            key_word = 'next_main_instrumentid'
        elif index == 0:
            key_word = 'previous_main_instrumentid'
        else:
            pass
        df = get_ts_data(tickers=tickers, start_date=start_date, end_date=end_date, key_word=key_word,
                         table_name='future.t_max_trade_vol_main_contract_map_daily', db_engine='quantanalysis',
                         code_str='main_contract_code', trade_dt_str='tradingday', code_str_filter='main_contract_code',
                         trade_dt_filter_str='tradingday', other_condition=None, as_matrix=False)
        df = df.dropna().reset_index(drop=True)
    else:
        raise ValueError("No roll method")
    return df


# 根据gsci合成的连续主力合约时间序列
def get_syn_con_ts(tickers, start_date, end_date, key_word, freq='D', index=1, ret_index=False, roll_key='gsci'):
    if roll_key == 'gsci':
        if freq == 'Daily' or freq == 'D':
            if index == 1:
                if ret_index:
                    if key_word == 'open':
                        key_word = 'open_return_pct'
                    elif key_word == 'close':
                        key_word = 'close_return_pct'
                    else:
                        key_word = 'settle_return_pct'
                    result = get_future_daily_data_qa(tickers=tickers, start_date=start_date, end_date=end_date,
                                                      key_word=key_word)
                    result.iloc[:, -1] = result.iloc[:, -1] / 100
                else:
                    result = get_future_daily_data_qa(tickers=tickers, start_date=start_date, end_date=end_date,
                                                      key_word=key_word)
            else:
                contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index)
                contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
                if ret_index:
                    ts_df = get_future_daily_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                     key_word=key_word, as_matrix=True)
                    ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                         columns=ts_df.columns)
                    ts_df['datetime'] = ts_df.index
                    ts_df = ts_df.melt(['datetime'])
                    ts_df = ts_df.iloc[:, [1, 0, 2]]

                else:
                    ts_df = get_future_daily_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                     key_word=key_word)
                result = pd.merge(left=ts_df, right=contract_df, how='left',
                                  left_on=[ts_df.columns[0], ts_df.columns[1]],
                                  right_on=[contract_df.columns[2], contract_df.columns[1]])
                result = result.dropna(subset=result.columns[-1], axis=0, how='any')
                result = result[result.columns[:3]]
        elif freq == 'Minute' or freq == 'Min' or freq == 'min':
            contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index)
            contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
            if type(key_word) is str:
                key_word_list = [key_word, 'tradingday']
            elif type(key_word) is list:
                key_word_list = key_word + ['tradingday']
            else:
                raise TypeError('key_word should be list or str')
            if ret_index:
                ts_df = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                  key_word=key_word_list, as_matrix=True)
                ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                     columns=ts_df.columns)
                ts_df['datetime'] = ts_df.index
                ts_df = ts_df.melt(['datetime'])
                ts_df = ts_df[[ts_df.columns[2], ts_df.columns[0], ts_df.columns[-1]]]
                ts_df1 = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                   key_word=key_word_list)
                ts_df = pd.merge(left=ts_df, right=ts_df1, how='left', left_on=[ts_df.columns[0], ts_df.columns[1]],
                                 right_on=[ts_df1.columns[0], ts_df1.columns[1]])
                ts_df = ts_df.iloc[:, [0, 1, 2, 4]]
            else:
                ts_df = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                  key_word=key_word_list)

            result = pd.merge(left=ts_df, right=contract_df, how='left', left_on=[ts_df.columns[0], ts_df.columns[3]],
                              right_on=[contract_df.columns[2], contract_df.columns[1]])
            result = result.dropna(subset=[result.columns[-1]], axis=0, how='any')
            result = result[result.columns[:3]]
        else:
            raise ValueError('Freq should be D or Min')
        if type(key_word) is str:
            result.columns = ['Code', 'Trade_DT', key_word]
        elif type(key_word) is list:
            result.columns = ['Code', 'Trade_DT'] + key_word
        else:
            raise TypeError('key_word should be list or str')
    elif roll_key == 'oi':
        if freq == 'Daily' or freq == 'D':
            if index == 1:
                if ret_index:
                    if key_word == 'open':
                        key_word = 'open_return_pct'
                    elif key_word == 'close':
                        key_word = 'close_return_pct'
                    else:
                        key_word = 'settle_return_pct'
                    result = get_future_daily_data_qa(tickers=tickers, start_date=start_date, end_date=end_date,
                                                      key_word=key_word)
                    result.iloc[:, -1] = result.iloc[:, -1] / 100
                else:
                    result = get_future_daily_data_qa(tickers=tickers, start_date=start_date, end_date=end_date,
                                                      key_word=key_word)
            else:
                contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index)
                contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
                if ret_index:
                    ts_df = get_future_daily_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                     key_word=key_word, as_matrix=True)
                    ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                         columns=ts_df.columns)
                    ts_df['datetime'] = ts_df.index
                    ts_df = ts_df.melt(['datetime'])
                    ts_df = ts_df.iloc[:, [1, 0, 2]]

                else:
                    ts_df = get_future_daily_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                     key_word=key_word)
                result = pd.merge(left=ts_df, right=contract_df, how='left',
                                  left_on=[ts_df.columns[0], ts_df.columns[1]],
                                  right_on=[contract_df.columns[2], contract_df.columns[1]])
                result = result.dropna(subset=result.columns[-1], axis=0, how='any')
                result = result[result.columns[:3]]
        elif freq == 'Minute' or freq == 'Min' or freq == 'min':
            contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index)
            contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
            if type(key_word) is str:
                key_word_list = [key_word, 'tradingday']
            elif type(key_word) is list:
                key_word_list = key_word + ['tradingday']
            else:
                raise TypeError('key_word should be list or str')
            if ret_index:
                ts_df = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                  key_word=key_word_list, as_matrix=True)
                ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                     columns=ts_df.columns)
                ts_df['datetime'] = ts_df.index
                ts_df = ts_df.melt(['datetime'])
                ts_df = ts_df[[ts_df.columns[2], ts_df.columns[0], ts_df.columns[-1]]]
                ts_df1 = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                   key_word=key_word_list)
                ts_df = pd.merge(left=ts_df, right=ts_df1, how='left', left_on=[ts_df.columns[0], ts_df.columns[1]],
                                 right_on=[ts_df1.columns[0], ts_df1.columns[1]])
                ts_df = ts_df.iloc[:, [0, 1, 2, 4]]
            else:
                ts_df = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                                  key_word=key_word_list)

            result = pd.merge(left=ts_df, right=contract_df, how='left', left_on=[ts_df.columns[0], ts_df.columns[3]],
                              right_on=[contract_df.columns[2], contract_df.columns[1]])
            result = result.dropna(subset=[result.columns[-1]], axis=0, how='any')
            result = result[result.columns[:3]]
        else:
            raise ValueError('Freq should be D or Min')
        if type(key_word) is str:
            result.columns = ['Code', 'Trade_DT', key_word]
        elif type(key_word) is list:
            result.columns = ['Code', 'Trade_DT'] + key_word
        else:
            raise TypeError('key_word should be list or str')
    else:
        raise TypeError('roll key should be gsci or oi')
    result = result.sort_values('Trade_DT')
    result.index = range(len(result))
    return result


# 连续主力合约时间序列
def get_continuous_future_ts(tickers, start_date, end_date, key_word, freq='halfDay', index=1, ret_index=False,
                             roll_method='gsci'):
    if freq == 'Daily' or freq == 'D':
        # if index == 1:
        #     if ret_index:
        #         if key_word == 'open':
        #             key_word = 'open_return_pct'
        #         elif key_word == 'close':
        #             key_word = 'close_return_pct'
        #         else:
        #             key_word = 'settle_return_pct'
        #         result = get_future_daily_data_qa(tickers=tickers, start_date=start_date, end_date=end_date,
        #                                           key_word=key_word)
        #         result.iloc[:, -1] = result.iloc[:, -1] / 100
        #     else:
        #         result = get_future_daily_data_qa(tickers=tickers, start_date=start_date, end_date=end_date,
        #                                           key_word=key_word)
        # else:
        contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index,
                                         roll_method=roll_method)
        contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
        if ret_index:
            ts_df = get_future_daily_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                             key_word=key_word, as_matrix=True)
            ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                 columns=ts_df.columns)
            ts_df['datetime'] = ts_df.index
            ts_df = ts_df.melt(['datetime'])
            ts_df = ts_df.iloc[:, [1, 0, 2]]

        else:
            ts_df = get_future_daily_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                             key_word=key_word)
        result = pd.merge(left=ts_df, right=contract_df, how='left',
                          left_on=[ts_df.columns[0], ts_df.columns[1]],
                          right_on=[contract_df.columns[2], contract_df.columns[1]])
        result = result.dropna(subset=result.columns[-1], axis=0, how='any')
        result = result[result.columns[:3]]
    elif freq in ['10min', '15min', '2min', '3min', '30min', '5min', 'halfDay']:
        contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index,
                                         roll_method=roll_method)
        contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
        if type(key_word) is str:
            key_word_list = [key_word, 'tradingday']
        elif type(key_word) is list:
            key_word_list = key_word + ['tradingday']
        else:
            raise TypeError('key_word should be list or str')
        freq_str = f"agg_frequency='{freq}'"
        if ret_index:
            ts_df = get_future_minute_aggregation_data_sm(tickers=contract_tickers, start_date=start_date,
                                                          end_date=end_date,
                                                          key_word=key_word_list, as_matrix=True,
                                                          other_condition=freq_str)
            ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                 columns=ts_df.columns)
            ts_df['datetime'] = ts_df.index
            ts_df = ts_df.melt(['datetime'])
            ts_df = ts_df[[ts_df.columns[2], ts_df.columns[0], ts_df.columns[-1]]]
            ts_df1 = get_future_minute_aggregation_data_sm(tickers=contract_tickers, start_date=start_date,
                                                           end_date=end_date,
                                                           key_word=key_word_list, other_condition=freq_str)
            ts_df = pd.merge(left=ts_df, right=ts_df1, how='left', left_on=[ts_df.columns[0], ts_df.columns[1]],
                             right_on=[ts_df1.columns[0], ts_df1.columns[1]])
            ts_df = ts_df.iloc[:, [0, 1, 2, 4]]
        else:
            ts_df = get_future_minute_aggregation_data_sm(tickers=contract_tickers, start_date=start_date,
                                                          end_date=end_date,
                                                          key_word=key_word_list, other_condition=freq_str)

        result = pd.merge(left=ts_df, right=contract_df, how='left', left_on=[ts_df.columns[0], ts_df.columns[3]],
                          right_on=[contract_df.columns[2], contract_df.columns[1]])
        result = result.dropna(subset=[result.columns[-1]], axis=0, how='any')
        result = result[result.columns[:3]]
    elif freq == 'Minute' or freq == 'Min' or freq == 'min' or freq == '1min':
        contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=index,
                                         roll_method=roll_method)
        contract_tickers = list(np.unique(contract_df[contract_df.columns[-1]]))
        if type(key_word) is str:
            key_word_list = [key_word, 'tradingday']
        elif type(key_word) is list:
            key_word_list = key_word + ['tradingday']
        else:
            raise TypeError('key_word should be list or str')
        if ret_index:
            ts_df = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                              key_word=key_word_list, as_matrix=True)
            ts_df = pd.DataFrame(data=ts_df.values[1:, :] / ts_df.values[:-1, :] - 1, index=ts_df.index[1:],
                                 columns=ts_df.columns)
            ts_df['datetime'] = ts_df.index
            ts_df = ts_df.melt(['datetime'])
            ts_df = ts_df[[ts_df.columns[2], ts_df.columns[0], ts_df.columns[-1]]]
            ts_df1 = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                               key_word=key_word_list)
            ts_df = pd.merge(left=ts_df, right=ts_df1, how='left', left_on=[ts_df.columns[0], ts_df.columns[1]],
                             right_on=[ts_df1.columns[0], ts_df1.columns[1]])
            ts_df = ts_df.iloc[:, [0, 1, 2, 4]]
        else:
            ts_df = get_future_minute_data_sm(tickers=contract_tickers, start_date=start_date, end_date=end_date,
                                              key_word=key_word_list)

        result = pd.merge(left=ts_df, right=contract_df, how='left', left_on=[ts_df.columns[0], ts_df.columns[3]],
                          right_on=[contract_df.columns[2], contract_df.columns[1]])
        result = result.dropna(subset=[result.columns[-1]], axis=0, how='any')
        result = result[result.columns[:3]]
    else:
        raise ValueError('Freq should be D or Min')
    if type(key_word) is str:
        result.columns = ['Code', 'Trade_DT', key_word]
    elif type(key_word) is list:
        result.columns = ['Code', 'Trade_DT'] + key_word
    else:
        raise TypeError('key_word should be list or str')
    return result


# 期货所有合约与品种映射(剔除连续与主力)
def get_code_instrument_mapping():
    sql_statement = 'SELECT ts_code,exchange,fut_code FROM future.t_instrument where per_unit is not null'
    data = sql_query(sql_statement=sql_statement, db_engine='sm')
    dict_map = {'CFFEX': 'CFE', 'CZCE': 'CZC', 'SHFE': 'SHF'}
    data = data.replace({'exchange': dict_map})
    result_map = {data.iloc[i, 0]: data.iloc[i, 2] + '.' + data.iloc[i, 1] for i in range(len(data))}
    return result_map


# 期货主力与此主力序列，主力与次主力日期mapping
def get_term_structure_series(tickers, start_date, end_date, key_word):
    main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=1)
    second_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=2)
    pair_df = pd.merge(left=main_contract_df, right=second_contract_df, how='inner',
                       on=['main_contract_code', 'tradingday'])
    # pair_df = pair_df.loc[:, ['main_contract_code', 'current_main_instrumentid',
    #                           'next_main_instrumentid']].drop_duplicates().reset_index(drop=True)

    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    df = get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word=key_word,
                                  as_matrix=True)

    return pair_df, df


# 主力与次主力合约价差或ratio
def get_pair_series(df, pair_df_all, method='spread'):
    pair_df = pair_df_all.loc[:, ['main_contract_code', 'current_main_instrumentid',
                                  'next_main_instrumentid']].drop_duplicates().reset_index(drop=True)
    column_list = [pair_df['current_main_instrumentid'][i] + '_' + pair_df['next_main_instrumentid'][i] for i in
                   range(len(pair_df))]
    result = pd.DataFrame(index=df.index, columns=column_list)

    price_df = pd.DataFrame(index=df.index, columns=column_list)
    for i in range(len(pair_df)):
        ticker1 = pair_df['current_main_instrumentid'][i]
        ticker2 = pair_df['next_main_instrumentid'][i]
        code = ticker1 + '_' + ticker2
        try:
            if method == 'spread':
                result[code] = df[ticker1] - df[ticker2]
                price_df[code] = df[ticker1] + df[ticker2]
            elif method == 'ratio':
                result[code] = df[ticker1] / df[ticker2]
                price_df[code] = df[ticker1] + df[ticker2]
            else:
                raise ValueError('method should be spread or ratio')
        except Exception as e:
            result[code] = np.nan

    return result, price_df


# 根据mapping表拼接成连续序列
def joint_con_ts(pair_df_all, pair_spread):
    # result = pd.DataFrame(index=pair_spread.index, columns=code_list)
    pair_df = pair_df_all.copy()
    pair_df['code'] = pair_df['current_main_instrumentid'] + '_' + pair_df['next_main_instrumentid']
    pair_spread = pair_spread.reset_index().melt('trade_date')
    pair_spread.columns = ['tradingday', 'code', 'value']
    result = pd.merge(left=pair_df, right=pair_spread, how='left', left_on=['code', 'tradingday'],
                      right_on=['code', 'tradingday'])
    result = result.pivot_table(index='tradingday', columns='main_contract_code', values='value')
    return result


# roll index
def roll_index(tickers, start_date, end_date, as_matrix=True):
    df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, key_word='roll_next_day',
                            index=0)
    df['roll_next_day'] = df['roll_next_day'].fillna(0)
    df = df.dropna()
    if as_matrix:
        df = df.pivot_table(index='tradingday', columns='main_contract_code', values='roll_next_day')
        df = df.fillna(0)
    return df


# vwap of current main instrument and next main instrument
def vwap_spread_table(tickers, start_date, end_date):
    roll_index_df = roll_index(tickers=tickers, start_date=start_date, end_date=end_date)
    pass


# 期货合约上市和到期日
def get_list_date_and_delist_date_of_instrument():
    sql_statement = 'select * from future.t_instrument'
    try:
        data = sql_query_from_sm(sql_statement=sql_statement)
        data = data[[data.columns[0], data.columns[-3], data.columns[-4]]]
    except Exception as e:
        time.sleep(1)
        data = sql_query_from_sm(sql_statement=sql_statement)
        data = data[[data.columns[0], data.columns[-3], data.columns[-4]]]
    return data


# 交易所交易时间
def get_trading_calendar(exchange=None):
    if exchange is None:
        exchange = 'SHF'
    exchange_map = ConstFutBasic.exchange_full_name_to_simple_name_mapping
    exchange_map = {value: key for key, value in exchange_map.items()}
    exchange_all_name = exchange_map[exchange]
    sql_statement = f'select cal_date from future.t_calendar where exchange=\'{exchange_all_name}\' and is_open=\'1\' order by cal_date'
    calendar_df = sql_query_from_sm(sql_statement=sql_statement)
    return calendar_df


# 提取期货主力与次主力合约时间序列
def get_fut_code_current_main_and_next_main_ts_data(tickers, start_date, end_date, key_word, as_matrix=False):
    main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=1)
    second_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=2)
    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    df = get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word=key_word,
                                  as_matrix=as_matrix)
    return df


# 提取期货主力与次主力合约时间序列vwap
def get_fut_code_current_main_and_next_main_vwap_matrix(tickers, start_date, end_date):
    main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=1)
    second_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=2)
    tickers_list = list(
        np.unique(np.append(np.array(main_contract_df.iloc[:, 2]), np.array(second_contract_df.iloc[:, 2]))))

    amount = get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='amount',
                                      as_matrix=False)
    vol = get_future_daily_data_sm(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='vol',
                                   as_matrix=False)
    data = pd.merge(left=vol, right=amount, how='inner', on=['ts_code', 'trade_date'])
    data['fut_code'] = data['ts_code'].map(get_code_instrument_mapping())
    data['bpv'] = data['fut_code'].map(ConstFutBasic.fut_code_bpv)
    data['vwap'] = data['amount'] * 10000 / data['vol'] / data['bpv']
    data = data[['ts_code', 'trade_date', 'vwap']]
    return data


# 提取gsci标准roll时间表
def get_gsci_roll_df(tickers, start_date, end_date):
    main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=1)
    second_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=2)
    pair_df = pd.merge(left=main_contract_df, right=second_contract_df, how='inner',
                       on=['main_contract_code', 'tradingday'])
    return pair_df


# 根据到期日roll
def get_delist_date_roll_df(tickers, start_date, end_date, non_current_month_commidity_index=False,
                            non_current_cffex_index=False):
    if type(tickers) is str:
        start_date_revise = str(int(start_date[:4]) - 1) + start_date[4:]
        end_date_revise = str(int(end_date[:4]) + 1) + end_date[4:]

        main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date_revise, end_date=end_date_revise,
                                              index=1)
        delist_date_df = get_list_date_and_delist_date_of_instrument()
        delist_date_df['main_contract_code'] = delist_date_df['ts_code'].map(get_code_instrument_mapping())
        delist_date_df = delist_date_df[['main_contract_code', 'delist_date', 'ts_code']]
        delist_date_df.columns = ['main_contract_code', 'tradingday', 'ts_code']
        delist_date_df = delist_date_df[
            delist_date_df['ts_code'].isin(
                np.unique(main_contract_df['current_main_instrumentid']))].reset_index(drop=True)
        main_code = np.array(delist_date_df.sort_values('ts_code')['ts_code'])
        main_code_dict = {main_code[i]: main_code[i + 1] for i in range(len(main_code) - 1)}
        main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=1)
        data = pd.merge(left=main_contract_df, right=delist_date_df, how='left',
                        on=['main_contract_code', 'tradingday'])
        data = data.sort_values(['main_contract_code', 'tradingday']).fillna(method='backfill').reset_index(drop=True)

        next_code = main_code_dict[data['ts_code'].unique()[-2]]
        # remain_na_code=np.array(data.loc[pd.isna(data['ts_code']),'current_main_instrumentid'])[0]
        data = data.fillna(next_code)

        data['fut_code'] = data['ts_code'].map(get_code_instrument_mapping())
        data['main'] = [
            data.loc[i, 'ts_code'] if data.loc[i, 'fut_code'] == data.loc[i, 'main_contract_code'] else data.loc[
                i, 'current_main_instrumentid'] for i in range(len(data))]
        pair_df = data[['main_contract_code', 'tradingday', 'main']]
        pair_df.columns = ['main_contract_code', 'tradingday', 'current_main_instrumentid']
        return pair_df
    elif type(tickers) is list:
        start_date_revise = str(int(start_date[:4]) - 1) + start_date[4:]
        end_date_revise = str(int(end_date[:4]) + 1) + end_date[4:]

        main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date_revise,
                                              end_date=end_date_revise,
                                              index=1)
        delist_date_df = get_list_date_and_delist_date_of_instrument()
        delist_date_df['main_contract_code'] = delist_date_df['ts_code'].map(get_code_instrument_mapping())
        delist_date_df = delist_date_df[['main_contract_code', 'delist_date', 'ts_code']]
        delist_date_df.columns = ['main_contract_code', 'tradingday', 'ts_code']
        delist_date_df = delist_date_df[
            delist_date_df['ts_code'].isin(
                np.unique(main_contract_df['current_main_instrumentid']))].reset_index(drop=True)

        main_code = np.array(delist_date_df.sort_values('ts_code')['ts_code'])
        main_code_dict = {main_code[i]: main_code[i + 1] for i in range(len(main_code) - 1)}
        main_contract_df = get_daily_contract(tickers=tickers, start_date=start_date, end_date=end_date, index=1)
        data = pd.merge(left=main_contract_df, right=delist_date_df, how='left',
                        on=['main_contract_code', 'tradingday'])
        data_all = pd.DataFrame()
        for t in tickers:
            d1 = data[data['main_contract_code'] == t]
            d1 = d1.sort_values(['main_contract_code', 'tradingday']).fillna(method='backfill').reset_index(
                drop=True)

            next_code = main_code_dict[d1['ts_code'].unique()[-2]]
            # remain_na_code=np.array(data.loc[pd.isna(data['ts_code']),'current_main_instrumentid'])[0]
            d1 = d1.fillna(next_code)
            data_all = pd.concat([data_all, d1])
        data_all['fut_code'] = data_all['ts_code'].map(get_code_instrument_mapping())
        data_all.index = range(len(data_all))
        data_all['main'] = [
            data_all.loc[i, 'ts_code'] if data_all.loc[i, 'fut_code'] == data_all.loc[i, 'main_contract_code'] else
            data_all.loc[
                i, 'current_main_instrumentid'] for i in range(len(data_all))]
        pair_df = data_all[['main_contract_code', 'tradingday', 'main']]
        pair_df.columns = ['main_contract_code', 'tradingday', 'current_main_instrumentid']
        pair_df.index = range(len(pair_df))

        if non_current_month_commidity_index or non_current_cffex_index:
            main_contract_mapping_dict = generate_main_contract_mapping()
            pair_df1 = pair_df[pair_df['main_contract_code'].isin(ConstFutBasic.group_cffex_stock)]
            pair_df1.index = range(len(pair_df1))
            pair_df2 = pair_df[~ pair_df['main_contract_code'].isin(ConstFutBasic.group_cffex_stock)]
            pair_df2.index = range(len(pair_df2))
            if non_current_month_commidity_index:
                pair_df2['Index1'] = pair_df2['tradingday'].apply(
                    lambda x: datetime.datetime.strftime(x, '%Y%m%d')[2:6])
                pair_df2['Index2'] = pair_df2['current_main_instrumentid'].apply(lambda x: x.split('.')[0][-4:])
                pair_df2['main'] = [
                    main_contract_mapping_dict[pair_df2.loc[i, 'current_main_instrumentid']] if pair_df2.loc[
                                                                                                    i, 'Index1'] ==
                                                                                                pair_df2.loc[
                                                                                                    i, 'Index2'] else
                    pair_df2.loc[i, 'current_main_instrumentid'] for i in range(len(pair_df2))]
                pair_df2 = pair_df2[['main_contract_code', 'tradingday', 'main']]
                pair_df2.columns = ['main_contract_code', 'tradingday', 'current_main_instrumentid']
            if non_current_cffex_index:
                pair_df1['Index1'] = pair_df1['tradingday'].apply(
                    lambda x: datetime.datetime.strftime(x, '%Y%m%d')[2:6])
                pair_df1['Index2'] = pair_df1['current_main_instrumentid'].apply(lambda x: x.split('.')[0][-4:])
                pair_df1['main'] = [
                    main_contract_mapping_dict[pair_df1.loc[i, 'current_main_instrumentid']] if pair_df1.loc[
                                                                                                    i, 'Index1'] ==
                                                                                                pair_df1.loc[
                                                                                                    i, 'Index2'] else
                    pair_df1.loc[i, 'current_main_instrumentid'] for i in range(len(pair_df1))]
                pair_df1 = pair_df1[['main_contract_code', 'tradingday', 'main']]
                pair_df1.columns = ['main_contract_code', 'tradingday', 'current_main_instrumentid']
            pair_df = pd.concat([pair_df1, pair_df2])
            pair_df = pair_df.sort_values(['main_contract_code', 'tradingday'])
        return pair_df
    else:
        raise TypeError('Input of tickers should be str or list')


# 拼接成连续合约
def combine_to_con_fut_code_ts(df, roll_df):
    df = df.reset_index().melt(df.index.name)
    df.columns = ['tradingday', 'code', 'value']
    result = pd.merge(left=roll_df, right=df, how='left', left_on=['current_main_instrumentid', 'tradingday'],
                      right_on=['code', 'tradingday'])
    result = result.pivot_table(index='tradingday', columns='main_contract_code', values='value')
    return result


# 提取cash basis
def get_cash_basis(tickers, start_date, end_date, key_word='cashbasis', table_name="future.t_marketdata_cash_basis",
                   db_engine='quantanalysis', code_str='productcode', trade_dt_str='datetime',
                   code_str_filter='productcode', trade_dt_filter_str='datetime', other_condition=None,
                   as_matrix=False):
    result = get_ts_data(tickers, start_date, end_date, key_word=key_word, table_name=table_name, db_engine=db_engine,
                         code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                         trade_dt_filter_str=trade_dt_filter_str, other_condition=other_condition, as_matrix=as_matrix)
    if as_matrix:
        result.index = [datetime.date(int(datetime.datetime.strftime(x, "%Y%m%d")[:4]),
                                      int(datetime.datetime.strftime(x, "%Y%m%d")[4:6]),
                                      int(datetime.datetime.strftime(x, "%Y%m%d")[-2:])) for x in result.index]
        result.index.name = trade_dt_str
    else:
        result[trade_dt_str] = result[trade_dt_str].apply(
            lambda x: datetime.date(int(datetime.datetime.strftime(x, "%Y%m%d")[:4]),
                                    int(datetime.datetime.strftime(x, "%Y%m%d")[4:6]),
                                    int(datetime.datetime.strftime(x, "%Y%m%d")[-2:])))
    return result


# 提取warehouse信息
def get_ware_house(tickers, start_date, end_date, key_word='warehouse_total',
                   table_name="future.t_warehouse_total_daily",
                   db_engine='securitymaster', code_str='windcode', trade_dt_str='tradingday',
                   code_str_filter='windcode', trade_dt_filter_str='tradingday', other_condition=None,
                   as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word=key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str, other_condition=other_condition, as_matrix=as_matrix)


# 根据主力合约到期日roll_index_df，生成TTM矩阵
def get_ttm(tickers, start_date, end_date, roll_index_df=None, as_matrix=True, roll_threshold=0,
            non_current_month_commidity_index=False, non_current_cffex_index=False):
    if roll_index_df is None:
        roll_index_df = get_delist_date_roll_df(tickers=tickers, start_date=start_date, end_date=end_date,
                                                non_current_month_commidity_index=non_current_month_commidity_index,
                                                non_current_cffex_index=non_current_cffex_index)

    list_delist_date_df = get_list_date_and_delist_date_of_instrument()
    list_delist_date_df = list_delist_date_df[['ts_code', 'delist_date']]
    df = pd.merge(left=roll_index_df, right=list_delist_date_df, how='left', left_on='current_main_instrumentid',
                  right_on='ts_code')
    df = df.dropna()
    df.index = range(len(df))
    calendar_df = get_trading_calendar()
    calendar_dict = {calendar_df.loc[i, 'cal_date']: i for i in range(len(calendar_df))}
    df['TTM'] = [calendar_dict[df['delist_date'][i]] - calendar_dict[df['tradingday'][i]] if df['delist_date'][
                                                                                                 i] in calendar_dict.keys() else round(
        (df['delist_date'][i] - df['tradingday'][i]).days * 5 / 7) for i in range(len(df))]
    df = df[['main_contract_code', 'tradingday', 'TTM']]
    if as_matrix:
        df = df.pivot_table(index='tradingday', columns='main_contract_code', values='TTM')
    return df


######################################################


######################################################
#
# constant future basic information refresh function
#
######################################################


# 提取当前所有期货的交易时间段（分钟级）
def const_fut_basic_fut_code_trading_min_time_dict(tradingday=None):
    """
    get future code and  trading time
    :param tradingday: string, format "%Y-%m-%d', e.g. '2022-08-01'
    :return: dict, {code:trading time}
            e.g. {'CU.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'],['13:30', '15:00']]}
    """
    if tradingday is None:
        tradingday = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
    sql_statement = f'SELECT DISTINCT main_contract_code from future.t_cm_gsci_marketdata_daily_ts where tradingday=\'{tradingday}\''
    code_list = sql_query(sql_statement=sql_statement, db_engine='qa')
    result = {}
    for i in range(len(code_list)):
        code = code_list.iloc[i, 0]
        exchg = code.split('.')[1]
        if exchg == 'COMDTY':
            pass
        else:
            data = get_syn_con_ts(tickers=code, start_date='2022-08-01', end_date='2022-08-01', key_word='close',
                                  freq='min', index=1, ret_index=False)
            data['time'] = data['Trade_DT'].apply(lambda x: str(x)[11:16])
            for j in range(len(data)):
                if j == 0:
                    data.loc[j, 'Delta'] = 10000000
                elif j == len(data) - 1:
                    data.loc[j, 'Delta'] = 10000000
                else:
                    data.loc[j, 'Delta'] = max((data.iloc[j + 1, 1] - data.iloc[j, 1]).seconds,
                                               (data.iloc[j, 1] - data.iloc[j - 1, 1]).seconds)
            data = data[data['Delta'] > 120]
            data.index = range(len(data))
            time_list = []
            for j in range(len(data)):
                if j % 2 == 0:
                    t = [data.iloc[j, 3]]
                else:
                    t.append(data.iloc[j, 3])
                    time_list.append(t)
            dict1 = {code: time_list}
            result = {**result, **dict1}
            print(code + ' is ok')
    return result


# 提取期货code与中文名的映射表
def const_fut_basic_fut_code_to_chinese_name_mapping():
    """
    a function to get the future code to its chines name
    :return: a dict
            e.g. :{'AL.SHF': '沪铝主力', 'CU.SHF': '沪铜主力',...}
    """
    # code name
    sql_statement = 'SELECT ts_code,name from future.t_instrument where symbol=fut_code order by exchange'
    code_list = sql_query(sql_statement=sql_statement, db_engine='sm')
    code_list.index = code_list['ts_code']
    code_list = code_list['name']
    code_list.to_dict()
    return code_list


# 提取期货品种上市时间
def const_fut_basic_fut_code_listing_date():
    """
    future product and its listing date mapping
    :return:  dict {code: listing date}
            e.g. {'CU.SHF':'2000-01-04'}
    """
    # qa earliest record date
    sql_statement = 'SELECT main_contract_code,min(tradingday) d from future.t_cm_gsci_marketdata_daily_ts group by main_contract_code order by d '
    research_start_date = sql_query(sql_statement=sql_statement, db_engine='qa')
    research_start_date.index = research_start_date.iloc[:, 0]
    research_start_date['d'] = research_start_date['d'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
    research_start_date = research_start_date[research_start_date.columns[1]]
    research_start_date = research_start_date.to_dict()
    return research_start_date


# 期货合约乘数
def const_fut_basic_bpv():
    """
    a function to get the big point value of the future contract:
        for CFFEX: BPV=multiplier
        for DCE, SHFE, CZCE: BPV = per_unit
            especially for JD.DCE: BPV = per_unit *2
    :return: a dict
        e.g.: {'A': 10.0, 'AG': 15.0, 'AL': 5.0,...}
    """
    sql_statement = "select fut_code,  sum(per_unit)/count(*),sum(multiplier)/count(*) c3 from future.t_instrument s1 where trade_unit is not null group by fut_code "
    data = sql_query_from_sm(sql_statement=sql_statement)
    data.columns = ['fut_code', 'per_unit', 'multiplier']
    sql_statement = "select fut_code,exchange from future.t_instrument s2 group by fut_code, exchange"
    data_code = sql_query_from_sm(sql_statement=sql_statement)
    data = pd.merge(left=data, right=data_code, how='inner', on='fut_code')
    data = data.replace(np.nan, 1)
    data['BPV'] = data['per_unit'] * data['multiplier']

    data.loc[data['fut_code'] == 'JD', 'BPV'] = data.loc[data['fut_code'] == 'JD', 'BPV'] * 2

    data = data.sort_values('fut_code')
    data.index = range(len(data))
    data['exchange'] = data['exchange'].replace(ConstFutBasic.exchange_full_name_to_simple_name_mapping)
    data['code'] = data['fut_code'] + '.' + data['exchange']
    data = {data['fut_code'][i]: data['BPV'][i] for i in range(len(data))}
    return data


######################################################

# test
# 根据roll的频率将期货分组
def fut_code_classification_by_roll():
    sql_statement = 'select * from future.t_cm_gsci_schedule'
    data = sql_query_from_qa(sql_statement=sql_statement)
    data = data.groupby(data.columns.to_list()).apply(lambda x: tuple(x.index))
    return [list(i) for i in list(data)]


# gsci标准roll合约映射
def gsci_roll_next_code():
    sql_statement = 'select * from future.t_cm_gsci_schedule'
    data = sql_query_from_qa(sql_statement=sql_statement)
    result = {}
    for i in range(len(data)):
        key1 = data.iloc[i, 0]
        values = np.unique(data.iloc[i, 1:])
        dict1 = {values[i]: values[i + 1] for i in range(len(values) - 1)}
        dict1[values[-1]] = values[0]
        result[key1] = dict1
    return result


def generate_main_contract_mapping(start_year=2000, end_year=2030, tickers_list=None):
    if tickers_list is None:
        tickers_list = ConstFutBasic.fut_code_list
    result_dict = {}
    for tickers in tickers_list:
        roll_dict = ConstFutBasic.fut_code_roll_instrument[tickers]
        name = tickers.split('.')[0]
        exchg = tickers.split('.')[1]
        for y in range(start_year, end_year + 1):
            for k, v in roll_dict.items():
                origin_year = str(y)[-2:]
                origin_tickers = name + origin_year + k + '.' + exchg
                new_year = str(y)[-2:] if int(k) < int(v) else str(y + 1)[-2:]
                new_tickers = name + new_year + v + '.' + exchg
                # if len(result_dict)==0:
                #     result_dict={origin_tickers, new_tickers}
                # else:
                result_dict.update({origin_tickers: new_tickers})
    return result_dict


#######################################################################################
#
# crypto
#
####################################################################################
# 从security master库中提取分钟频率时间序列
def get_crypto_hour_data_sm(tickers, start_date, end_date, key_word, table_name="crypto.t_marketdata_hourly",
                            db_engine='securitymaster', code_str='symbol', trade_dt_str='datetime',
                            code_str_filter='symbol', trade_dt_filter_str='date',
                            other_condition=' exchange=\'binance\'', as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str,
                       other_condition=other_condition, as_matrix=as_matrix)


# 从security master库中提取日频时间序列
def get_crypto_daily_data_sm(tickers, start_date, end_date, key_word, table_name="crypto.t_marketdata_daily",
                             db_engine='securitymaster', code_str='symbol', trade_dt_str='datetime',
                             code_str_filter='symbol', trade_dt_filter_str='date',
                             other_condition='exchange=\'binance\'', as_matrix=False):
    return get_ts_data(tickers, start_date, end_date, key_word, table_name=table_name, db_engine=db_engine,
                       code_str=code_str, trade_dt_str=trade_dt_str, code_str_filter=code_str_filter,
                       trade_dt_filter_str=trade_dt_filter_str,
                       other_condition=other_condition, as_matrix=as_matrix)


# 提取数字货币交易所的所有品种
def get_crypto_tickers_list(exchange=None, table_name="crypto.t_marketdata_hourly"):
    if exchange is None:
        sql_statement = f"select distinct symbol from {table_name} "
    else:
        sql_statement = f"select distinct symbol from {table_name} where exchange=\'{exchange}\'"
    data = sql_query(sql_statement=sql_statement, db_engine='sm')
    try:
        data = list(data['symbol'])
    except KeyError:
        data = []
    return data

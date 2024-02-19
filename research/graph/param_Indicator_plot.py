import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

import data.ConstantData.future_basic_information as ConstFutBasic
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import data.Process.fut_data_process as DataProcessFut


def one_dimension_param_one_dimension_indicator(data, file_location=None):
    df = data.copy()
    code_list = data['code'].unique()
    param1_list = data['window'].unique()
    param2_list = data['window_yoy'].unique()
    param3_list = data['start'].unique()
    param4_list = data['end'].unique()
    for code in code_list:
        df11 = df[df['code'] == code]
        for yoy in param2_list:
            df1 = df11[df11['window_yoy'] == yoy]
            fig, axs = plt.subplots(len(param3_list), len(param4_list),
                                    figsize=(5 * len(param3_list), 5 * len(param4_list)), sharey=True)
            for i in range(len(param3_list)):
                start_num = param3_list[i]
                for j in range(len(param4_list)):
                    end_num = param4_list[j]
                    str_title = 'start: ' + str(start_num) + '  end: ' + str(end_num)
                    df_ij = df1[df1['start'] == start_num]
                    df_ij = df_ij[df_ij['end'] == end_num]
                    df_ir = df_ij[df_ij['variable'] == 'IR'][['window', 'value']].sort_values('window')
                    df_calmar = df_ij[df_ij['variable'] == 'Calmar'][['window', 'value']].sort_values('window')
                    axs[i, j].plot(df_ir.iloc[:, 0], df_ir.iloc[:, 1], label=['IR'])
                    axs[i, j].plot(df_calmar.iloc[:, 0], df_calmar.iloc[:, 1], label=['Calmar'])
                    axs[i, j].set_title(str_title)
                    axs[i, j].legend(loc='upper right')
            plt.savefig(
                'C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\graph\\' + code + '_yoy_' + str(
                    yoy) + '.png')
            plt.close()

    df = data.copy()
    code_list = data['code'].unique()
    param1_list = data['window'].unique()
    param2_list = data['window_yoy'].unique()
    param3_list = data['start'].unique()
    param4_list = data['end'].unique()

    for start_num in param3_list:
        df11 = df[df['start'] == start_num]
        for end_num in param4_list:
            df1 = df11[df11['end'] == end_num]
            fig, axs = plt.subplots(len(code_list), len(param2_list),
                                    figsize=(5 * len(code_list), 5 * len(param2_list)))
            for i in range(len(code_list)):
                code = code_list[i]
                for j in range(len(param2_list)):
                    yoy = param2_list[j]
                    str_title = 'code: ' + str(code) + '  yoy: ' + str(yoy)
                    df_ij = df1[df1['code'] == code]
                    df_ij = df_ij[df_ij['window_yoy'] == yoy]
                    df_ir = df_ij[df_ij['variable'] == 'IR'][['window', 'value']].sort_values('window')
                    df_calmar = df_ij[df_ij['variable'] == 'Calmar'][['window', 'value']].sort_values('window')
                    axs[i, j].plot(df_ir.iloc[:, 0], df_ir.iloc[:, 1], label=['IR'])
                    axs[i, j].plot(df_calmar.iloc[:, 0], df_calmar.iloc[:, 1], label=['Calmar'])
                    axs[i, j].set_title(str_title)
                    axs[i, j].legend(loc='upper right')
            plt.savefig(
                'C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\graph\\start_' + str(
                    start_num) + '_end_' + str(
                    end_num) + '.png')
            plt.close()

    df = data.copy()
    code_list = data['code'].unique()
    param3_list = data['end'].unique()
    fig, axs = plt.subplots(len(code_list), len(param3_list), figsize=(20, 90))
    for i in range(len(code_list)):
        code = code_list[i]
        df1 = df[df['code'] == code]
        for j in range(len(param3_list)):
            end_num = param3_list[j]
            window_yoy = 3
            str_title = code + ' end: ' + str(end_num)
            df_ij = df1[df1['end'] == end_num]
            df_ij = df_ij[df_ij['window_yoy'] == window_yoy]
            df_ir = df_ij[df_ij['variable'] == 'IR'][['window', 'value']].sort_values('window')
            df_calmar = df_ij[df_ij['variable'] == 'Calmar'][['window', 'value']].sort_values('window')
            axs[i, j].plot(df_ir.iloc[:, 0], df_ir.iloc[:, 1], label=['IR'])
            axs[i, j].plot(df_calmar.iloc[:, 0], df_calmar.iloc[:, 1], label=['Calmar'])
            axs[i, j].set_title(str_title)
            axs[i, j].legend(loc='upper right')
    plt.savefig('C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\graph\\all.png')
    plt.close()


def parameter_pnl_group(data, file_path=None):
    if file_path is None:
        file_path = 'C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\param_test\\'
    code_list = data['code'].unique()
    param1_list = np.sort(data['window'].unique())
    param2_list = np.sort(data['window_yoy'].unique())
    param3_list = np.sort(data['start'].unique())
    param4_list = np.sort(data['end'].unique())

    param1_list = [3, 6, 12, 30]

    for start_date1 in param3_list:
        for end_date1 in param4_list:
            df = data[data['start'] == start_date1]
            df = df[df['end'] == end_date1]
            data_all = pd.DataFrame()
            graph_name = 'start_date_' + str(start_date1) + '_end_date_' + str(end_date1) + 'cumpnl.png'
            graph_location = 'C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\graph\\' + graph_name
            fig, axs = plt.subplots(len(code_list), len(param2_list),
                                    figsize=(8 * len(code_list), 8 * len(param2_list)))
            for i in range(len(code_list)):
                code = code_list[i]
                for j in range(len(param2_list)):
                    window_yoy = param2_list[j]
                    title = code + ' window_yoy: ' + str(window_yoy)
                    data_code_param_all = pd.DataFrame()
                    for window in param1_list:
                        try:
                            file_name = f"result_4factor_{window}_{window_yoy}_m1_{start_date1}_m2_{end_date1}.xlsx"
                            file_location = file_path + file_name
                            data_code_param = pd.read_excel(file_location, sheet_name='sector_cum_pnl_no_fee',
                                                            index_col=0)
                            data_code_param = data_code_param[code]
                            # data_code_param.name=['window: '+str(window)]
                            data_code_param_all = pd.concat([data_code_param_all, data_code_param], axis=1)
                        except Exception as e:
                            data_code_param = pd.DataFrame(index=data_code_param_all, columns=code)
                            data_code_param_all = pd.concat([data_code_param_all, data_code_param], axis=1)
                    data_code_param_all.columns = ['window: ' + str(window) for window in param1_list]
                    for k in range(len(data_code_param_all.columns)):
                        axs[i, j].plot(data_code_param_all.index, data_code_param_all.iloc[:, k],
                                       label=data_code_param_all.columns[k])
                    axs[i, j].set_title(title)
                    axs[i, j].legend(loc='upper left')
                    print(code + str(window_yoy) + ' is ok')
            plt.savefig(graph_location)
            plt.close()


def draw_intraday_oi_correlation():
    theshold_num = 60
    data = pd.read_csv(
        r'C:\Users\jason.huang\research\factor_data\OI_change\intraday_OI_correlation_4_types\data_all\intraday_OI_correlation_all.csv',
        index_col=0)
    data_p = pd.read_csv(
        r'C:\Users\jason.huang\research\factor_data\OI_change\intraday_OI_correlation_4_types\data_all\intraday_OI_correlation_pvalue_all.csv',
        index_col=0)

    code_list = np.unique(data['code'])
    data['year'] = data['main'].apply(lambda x: x.split('.')[0][-4:-2])
    data['month'] = data['main'].apply(lambda x: x.split('.')[0][-2:])
    data['tradingday'] = data['tradingday'].apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
    save_path = 'C:\\Users\\jason.huang\\research\\factor_data\OI_change\\intraday_OI_correlation_4_types\\graph\\'
    for code in code_list:
        df1 = data[data['code'] == code]
        m = len(df1['year'].unique())
        n = len(df1['month'].unique())
        save_location = save_path + code + '.png'
        fig, axs = plt.subplots(n, m, figsize=(10 * m, 5 * n))
        for i in range(m):
            for j in range(n):
                year = np.sort(df1['year'].unique())[i]
                month = np.sort(df1['month'].unique())[j]

                df_data = df1[df1['year'] == year]
                df_data = df_data[df_data['month'] == month]
                df_data = df_data[['main', 'tradingday', 'correlation']]
                if len(df_data) == 0:
                    pass
                else:
                    if len(df_data) > theshold_num:
                        df_data = df_data.iloc[(len(df_data) - theshold_num):len(df_data), :]
                    else:
                        pass
                    axs[j, i].bar(df_data['tradingday'], df_data['correlation'])
                    axs[j, i].plot(df_data['tradingday'], np.array([0] * len(df_data)), color='black')
                    try:
                        axs[j, i].plot(np.array(df_data['tradingday'][-10], df_data['tradingday'][-10]),
                                       np.array([-0.5, 0.5]), color='red')
                        axs[j, i].plot(np.array(df_data['tradingday'][-40], df_data['tradingday'][-40]),
                                       np.array([-0.5, 0.5]), color='red')
                    except Exception as e:
                        pass
                    try:
                        axs[j, i].axhline(x=len(df_data) - 10, color='red')
                        axs[j, i].axhline(x=len(df_data) - 40, color='red')
                    except Exception as e:
                        pass
                    axs[j, i].set_title(np.array(df_data['main'])[0])

        plt.savefig(save_location)
        plt.close()
        print(code + ' is ok')


def draw_intraday_oi_correlation_4_type():
    theshold_num = 60
    data = pd.read_csv(
        r'C:\Users\jason.huang\research\factor_data\OI_change\intraday_OI_correlation_4_types\data_all\intraday_OI_correlation_all.csv',
        index_col=0)
    data_p = pd.read_csv(
        r'C:\Users\jason.huang\research\factor_data\OI_change\intraday_OI_correlation_4_types\data_all\intraday_OI_correlation_pvalue_all.csv',
        index_col=0)

    code_list = np.unique(data['code'])
    data['year'] = data['main'].apply(lambda x: x.split('.')[0][-4:-2])
    data['month'] = data['main'].apply(lambda x: x.split('.')[0][-2:])
    data['tradingday'] = data['tradingday'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    save_path = 'C:\\Users\\jason.huang\\research\\factor_data\OI_change\\intraday_OI_correlation_4_types\\graph\\'
    corr_list = ['ic', 'rankic', 'ic_small_num',
                 'rankic_small_num', 'ic_large_num', 'rankic_large_num', 'ic_intraday',
                 'rankic_intraday', 'ic_open_close', 'rankic_open_close', 'ic_oi_shaped',
                 'rankic_oi_shaped']

    for code in code_list:
        for corr_name in corr_list:
            df1 = data[data['code'] == code]
            m = len(df1['year'].unique())
            n = len(df1['month'].unique())
            save_location = save_path + code + '_' + corr_name + '.png'
            fig, axs = plt.subplots(n, m, figsize=(10 * m, 5 * n))
            for i in range(m):
                for j in range(n):
                    year = np.sort(df1['year'].unique())[i]
                    month = np.sort(df1['month'].unique())[j]

                    df_data = df1[df1['year'] == year]
                    df_data = df_data[df_data['month'] == month]
                    # df_data = df_data[['main', 'tradingday', 'correlation']]
                    df_data = df_data[['main', 'tradingday', corr_name]]
                    df_data.columns = ['main', 'tradingday', 'correlation']

                    if len(df_data) == 0:
                        pass
                    else:
                        if len(df_data) > theshold_num:
                            df_data = df_data.iloc[(len(df_data) - theshold_num):len(df_data), :]
                        else:
                            pass
                        df_data.index = range(len(df_data))
                        axs[j, i].bar(pd.to_datetime(df_data['tradingday']), df_data['correlation'])
                        axs[j, i].plot(pd.to_datetime(df_data['tradingday']), np.array([0] * len(df_data)),
                                       color='black')
                        try:
                            axs[j, i].plot([pd.to_datetime(df_data.loc[len(df_data) - 10, 'tradingday']),
                                            pd.to_datetime(df_data.loc[len(df_data) - 10, 'tradingday'])], [-0.5, 0.5],
                                           color='r',
                                           linestyle='--')
                            axs[j, i].plot([pd.to_datetime(df_data.loc[len(df_data) - 40, 'tradingday']),
                                            pd.to_datetime(df_data.loc[len(df_data) - 40, 'tradingday'])], [-0.5, 0.5],
                                           color='r',
                                           linestyle='--')
                        except Exception as e:
                            pass
                        axs[j, i].set_title(np.array(df_data['main'])[0])

            plt.savefig(save_location)
            plt.close()
            print(code + '  ' + corr_name + ' is ok')

    # p-value graph
    theshold_num = 60
    data = pd.read_csv(
        r'C:\Users\jason.huang\research\factor_data\OI_change\intraday_OI_correlation_4_types\data_all\intraday_OI_correlation_all.csv',
        index_col=0)
    data_p = pd.read_csv(
        r'C:\Users\jason.huang\research\factor_data\OI_change\intraday_OI_correlation_4_types\data_all\intraday_OI_correlation_pvalue_all.csv',
        index_col=0)
    data_p.iloc[:, 4:] = data_p.iloc[:, 4:].applymap(lambda x: 1 if x < 0.1 else 0)

    code_list = np.unique(data['code'])
    data['year'] = data['main'].apply(lambda x: x.split('.')[0][-4:-2])
    data['month'] = data['main'].apply(lambda x: x.split('.')[0][-2:])
    data['tradingday'] = data['tradingday'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    save_path = 'C:\\Users\\jason.huang\\research\\factor_data\OI_change\\intraday_OI_correlation_4_types\\graph_p_value_correct\\'
    corr_list = ['ic', 'rankic', 'ic_small_num',
                 'rankic_small_num', 'ic_large_num', 'rankic_large_num', 'ic_intraday',
                 'rankic_intraday', 'ic_open_close', 'rankic_open_close', 'ic_oi_shaped',
                 'rankic_oi_shaped']

    for code in code_list:
        for corr_name in corr_list:
            df1 = data[data['code'] == code]
            m = len(df1['year'].unique())
            n = len(df1['month'].unique())
            save_location = save_path + code + '_' + corr_name + '.png'
            fig, axs = plt.subplots(n, m, figsize=(10 * m, 5 * n))
            for i in range(m):
                for j in range(n):
                    year = np.sort(df1['year'].unique())[i]
                    month = np.sort(df1['month'].unique())[j]

                    df_data = df1[df1['year'] == year]
                    df_data = df_data[df_data['month'] == month]
                    # df_data = df_data[['main', 'tradingday', 'correlation']]
                    df_data = df_data[['main', 'tradingday', corr_name]]
                    df_data.columns = ['main', 'tradingday', 'correlation']

                    if len(df_data) == 0:
                        pass
                    else:
                        if len(df_data) > theshold_num:
                            df_data = df_data.iloc[(len(df_data) - theshold_num):len(df_data), :]
                        else:
                            pass
                        df_data.index = range(len(df_data))
                        axs[j, i].bar(pd.to_datetime(df_data['tradingday']), df_data['correlation'])
                        axs[j, i].plot(pd.to_datetime(df_data['tradingday']), np.array([0] * len(df_data)),
                                       color='black')
                        try:
                            axs[j, i].plot([pd.to_datetime(df_data.loc[len(df_data) - 10, 'tradingday']),
                                            pd.to_datetime(df_data.loc[len(df_data) - 10, 'tradingday'])], [-0.5, 0.5],
                                           color='r',
                                           linestyle='--')
                            axs[j, i].plot([pd.to_datetime(df_data.loc[len(df_data) - 40, 'tradingday']),
                                            pd.to_datetime(df_data.loc[len(df_data) - 40, 'tradingday'])], [-0.5, 0.5],
                                           color='r',
                                           linestyle='--')
                        except Exception as e:
                            pass
                        axs[j, i].set_title(np.array(df_data['main'])[0])

            plt.savefig(save_location)
            plt.close()
            print(code + '  ' + corr_name + ' is ok')


def draw_parameter_pnl_group(data, file_path=None):
    if file_path is None:
        file_path = 'C:\\Users\\jason.huang\\research\\backtest\single_factor_test\\cashbasis_front_contract_test\\result\\'
    code_list = data['code'].unique()
    param1_list = np.sort(data['ttm'].unique())
    param2_list = np.sort(data['factor'].unique())

    for start_date1 in param3_list:
        for end_date1 in param4_list:
            df = data[data['start'] == start_date1]
            df = df[df['end'] == end_date1]
            data_all = pd.DataFrame()
            graph_name = 'start_date_' + str(start_date1) + '_end_date_' + str(end_date1) + 'cumpnl.png'
            graph_location = 'C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\graph\\' + graph_name
            fig, axs = plt.subplots(len(code_list), len(param2_list),
                                    figsize=(8 * len(code_list), 8 * len(param2_list)))
            for i in range(len(code_list)):
                code = code_list[i]
                for j in range(len(param2_list)):
                    window_yoy = param2_list[j]
                    title = code + ' window_yoy: ' + str(window_yoy)
                    data_code_param_all = pd.DataFrame()
                    for window in param1_list:
                        try:
                            file_name = f"result_4factor_{window}_{window_yoy}_m1_{start_date1}_m2_{end_date1}.xlsx"
                            file_location = file_path + file_name
                            data_code_param = pd.read_excel(file_location, sheet_name='sector_cum_pnl_no_fee',
                                                            index_col=0)
                            data_code_param = data_code_param[code]
                            # data_code_param.name=['window: '+str(window)]
                            data_code_param_all = pd.concat([data_code_param_all, data_code_param], axis=1)
                        except Exception as e:
                            data_code_param = pd.DataFrame(index=data_code_param_all, columns=code)
                            data_code_param_all = pd.concat([data_code_param_all, data_code_param], axis=1)
                    data_code_param_all.columns = ['window: ' + str(window) for window in param1_list]
                    for k in range(len(data_code_param_all.columns)):
                        axs[i, j].plot(data_code_param_all.index, data_code_param_all.iloc[:, k],
                                       label=data_code_param_all.columns[k])
                    axs[i, j].set_title(title)
                    axs[i, j].legend(loc='upper left')
                    print(code + str(window_yoy) + ' is ok')
            plt.savefig(graph_location)
            plt.close()


def one_dimension_param_one_dimension_indicator2(data, file_location=None):
    df = data.copy()
    code_list = data['code'].unique()
    param1_list = data['ttm'].unique()
    factor_list = data['factor'].unique()

    fig, axs = plt.subplots(len(code_list), len(factor_list), figsize=(10 * len(factor_list), 10 * len(code_list)))
    for i in range(len(factor_list)):
        for j in range(len(code_list)):
            code = code_list[j]
            factor = factor_list[i]
            df1 = df[df['code'] == code]
            df1 = df1[df1['factor'] == factor]
            df1 = df1.sort_values('variable')
            labels = df1['variable'].unique()

            sorts = df1['ttm'].unique()
            a1 = np.array(df1[df1['ttm'] == sorts[0]]['value'])
            a2 = np.array(df1[df1['ttm'] == sorts[1]]['value'])
            width = 0.3
            x = np.arange(len(labels))
            axs[j, i].bar(x - width / 2, a1, width, label=sorts[0])
            axs[j, i].bar(x + width / 2, a2, width, label=sorts[1])

            title = code + '  ' + factor
            axs[j, i].set_title(title)
            axs[j, i].set_xticks(x, labels)
            axs[j, i].legend()

    plt.savefig(r'C:\Users\jason.huang\research\backtest\single_factor_test\all_factor\result.png')
    plt.close()


def draw_cumpnl_graph():
    pass


if __name__ == '__main__':
    data = pd.read_csv('C:\\Users\\jason.huang\\research\\backtest\\Event_Driven\\OI_change_rate\\param_all.csv',
                       index_col=0).reset_index(drop=True)
    data = data[data['start'] == 40]

    code_list = data['code'].unique()
    param1_list = data['window'].unique()
    param2_list = data['window_yoy'].unique()
    param3_list = data['start'].unique()
    param4_list = data['end'].unique()

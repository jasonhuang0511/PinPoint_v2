import os
import pandas as pd
import numpy as np


# 各个参数result拼接一起
def joint_file_together():
    file_path = 'C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\cashbasis_front_contract_test\\result\\'
    result = pd.DataFrame()
    for i in range(len(os.listdir(file_path))):
        file_name = os.listdir(file_path)[i]
        file_location = file_path + file_name
        data = pd.read_excel(file_location, 'sector_Bt_Indicator_no_fee', index_col=0)
        df_cum = pd.read_excel(file_location, 'sector_cum_pnl_no_fee', index_col=0)
        df_cum = df_cum.iloc[-1, :].reset_index()
        df_cum.columns = ['code', 'pnl']
        data = data.loc[:, ['code', 'IR', 'drawdown']]
        data = pd.merge(left=data, right=df_cum, how='left', on='code')
        data['IR'] = data['IR'] * 15.8
        data['Calmar'] = data['pnl'] / np.abs(data['drawdown'])
        data = data.loc[:, ['code', 'IR', 'Calmar']]

        data = data.melt('code')
        data['ttm']=file_name.split('.')[0].split('_')[-1]
        data['factor']='_'.join(file_name.split('.')[0].split('_')[:-1])
        if i == 0:
            result = data
        else:
            result = pd.concat([result, data])
        print(file_location + ' is ok')
    result.to_csv('C:\\Users\\jason.huang\\research\\backtest\\single_factor_test\\cashbasis_front_contract_test\\result_all.csv')
    return result

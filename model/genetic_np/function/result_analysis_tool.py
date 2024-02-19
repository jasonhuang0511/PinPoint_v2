import pickle
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_1samp
from scipy.stats import norm


def ret_t_test(a):
    return ttest_1samp(a, popmean=0, nan_policy='omit', alternative='greater').pvalue


class NetValueAnalysis:
    def __init__(self, nv):
        self.nv = nv
        self.stats = self._calculate_stats()

    def _calculate_stats(self):
        def __calculate_annual_return(ret):
            return ret.mean() * 252

        def __calculate_annual_ir(ret):
            return ret.mean() / (ret.std() + 0.0001) * 15.8

        def __calculate_calmar(ret):
            mdd = (ret.cumsum() - ret.cumsum().cummax()).min()
            return ret.sum() / (np.abs(mdd) + 0.001)

        stats = pd.DataFrame()
        stats.loc['Total', 'Ret'] = __calculate_annual_return(self.nv.diff())
        stats.loc['Total', 'IR'] = __calculate_annual_ir(self.nv.diff())
        stats.loc['Total', 'Calmar'] = __calculate_calmar(self.nv.diff())

        return stats

    def plot_strategy_nv(self):
        import matplotlib.pyplot as plt
        stats_show = self.stats.copy()
        for i in range(len(stats_show)):
            stats_show.iloc[i, 0] = str(round(stats_show.iloc[i, 0] * 100, 2)) + '%'
            stats_show.iloc[i, 1] = str(round(stats_show.iloc[i, 1], 2))
            stats_show.iloc[i, 2] = str(round(stats_show.iloc[i, 2], 2))
        fig, axs = plt.subplots(3, 1, figsize=(20, 30))

        axs[0].axis('off')
        the_table = axs[0].table(cellText=stats_show.values, colLabels=stats_show.columns, rowLabels=stats_show.index,
                                 cellLoc='center', loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(20)
        the_table.scale(1, 4)

        axs[1].plot(self.nv.fillna(method='ffill').fillna(0), label='strategy')
        axs[1].set_title("strategy nv and undetlying nv")

        axs[2].plot(self.nv - self.nv.cummax(), lw=3, label='drawdown', alpha=0.5)
        axs[2].fill_between(x=self.nv.index, y1=np.array([0] * len(self.nv)),
                            y2=self.nv - self.nv.cummax(), alpha=0.7)
        axs[2].legend()
        axs[2] = axs[2].twinx()
        axs[2].plot(
            self.nv.diff().rolling(60).mean() / self.nv.diff().rolling(60).std(),
            label='rolling sharpe', color='red', alpha=0.5)
        axs[2].legend()
        axs[2].set_title('Drawdown and Rolling Sharpe')


class StrategyIndicator:
    def __init__(self, ret, weight, out_of_sample_date, freq):
        self.ret = ret
        self.weight = weight
        # self.price = price
        self.freq = freq
        self.out_of_sample_date = out_of_sample_date
        self.stats = self._calculate_stas(ret=self.ret, weight=self.weight, out_of_sample_date=self.out_of_sample_date,
                                          freq=self.freq)

    @staticmethod
    def _calculate_stas(ret, weight, out_of_sample_date, freq):
        result = pd.DataFrame(index=['in_sample', 'out_of_sample', 'total'],
                              columns=['Ret', 'IR', 'Calmar', 'Turnover'])
        index = np.where(ret.index >= pd.to_datetime(out_of_sample_date))[0][0]
        # in sample
        ret1 = ret.iloc[:index, ]
        weight1 = weight.iloc[:index, ]
        result.loc['in_sample', 'Ret'] = ret1.mean() * 252 * freq
        result.loc['in_sample', 'IR'] = ret1.mean() / (ret1.std() + 0.0001) * np.sqrt(252 * freq)
        mdd1 = np.abs((ret1.cumsum() - ret1.cumsum().cummax()).min())
        result.loc['in_sample', 'IR'] = ret1.mean() / (mdd1 + 0.0001) * 252
        result.loc['in_sample', 'Turnover'] = (weight1.diff().abs().sum() / len(weight1)).mean()

        # out_of_sample
        ret1 = ret.iloc[index:, ]
        weight1 = weight.iloc[index:, ]
        result.loc['out_of_sample', 'Ret'] = ret1.mean() * 252 * freq
        result.loc['out_of_sample', 'IR'] = ret1.mean() / (ret1.std() + 0.0001) * np.sqrt(252 * freq)
        mdd1 = np.abs((ret1.cumsum() - ret1.cumsum().cummax()).min())
        result.loc['out_of_sample', 'IR'] = ret1.mean() / (mdd1 + 0.0001) * 252
        result.loc['out_of_sample', 'Turnover'] = (weight1.diff().abs().sum() / len(weight1)).mean()

        # total
        ret1 = ret.iloc[:, ]
        weight1 = weight.iloc[:, ]
        result.loc['total', 'Ret'] = ret1.mean() * 252 * freq
        result.loc['total', 'IR'] = ret1.mean() / (ret1.std() + 0.0001) * np.sqrt(252 * freq)
        mdd1 = np.abs((ret1.cumsum() - ret1.cumsum().cummax()).min())
        result.loc['total', 'IR'] = ret1.mean() / (mdd1 + 0.0001) * 252
        result.loc['total', 'Turnover'] = (weight1.diff().abs().sum() / len(weight1)).mean()

        return result


def read_single_pickle(file_location):
    with open(file_location, 'rb') as file:
        data = pickle.load(file)
    return data


def single_pickle_to_df(single_pickle):
    nv_single = pd.DataFrame(single_pickle.strategy_nv)
    nv_single.columns = [single_pickle.formula]
    return nv_single


def single_pickle_weight_to_dict(single_pickle):
    result_dict = {single_pickle.formula: pd.DataFrame(single_pickle.factor_weighting_value.fillna(0)).values}
    return result_dict
    # nv_single=pd.DataFrame(single_pickle.weight).values
    # nv_single.columns=[single_pickle.formula]
    # return nv_single


def delete_file_folder(delete_folder_absolute_path):
    os.system(f"rm -rf {delete_folder_absolute_path}")
    print(f"{delete_folder_absolute_path} is deleted")


def delete_non_use_pickle(file_path):
    pickle_location = file_path + 'total.pkl'
    with open(pickle_location, 'rb') as file:
        data = pickle.load(file)
    indicator_all = data.copy().reset_index(drop=True)
    indicator_all = indicator_all.drop_duplicates(['Ret_total', 'IR_total', 'Calmar_total'])
    selected_indicator = indicator_all[indicator_all['IR_train'] > 1]
    selected_indicator = selected_indicator[selected_indicator['Calmar_test'] > 0.5]
    delete_pickle_df = data.loc[[x for x in data.index if x not in selected_indicator.index], :]
    delete_pickle_location_list = list(delete_pickle_df['save_location'])
    for delete_pickle_location in delete_pickle_location_list:
        os.system(f"rm {delete_pickle_location}")
        print(delete_pickle_location)


def rolling_30_t_test_p_value(df):
    return df.rolling(30).apply(ret_t_test)


def rolling_60_t_test_p_value(df):
    return df.rolling(60).apply(ret_t_test)


def rolling_90_t_test_p_value(df):
    return df.rolling(90).apply(ret_t_test)


def rolling_120_t_test_p_value(df):
    return df.rolling(120).apply(ret_t_test)


def rolling_240_t_test_p_value(df):
    return df.rolling(240).apply(ret_t_test)


def rolling_30_t_test_p_value_parallel(df_list, n=None):
    if n is None:
        n = min(100, len(df_list) // 50 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(rolling_30_t_test_p_value, df_list, chunksize=10)
    return df_list


def rolling_60_t_test_p_value_parallel(df_list, n=None):
    if n is None:
        n = min(100, len(df_list) // 60 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(rolling_60_t_test_p_value, df_list, chunksize=10)
    return df_list


def rolling_90_t_test_p_value_parallel(df_list, n=None):
    if n is None:
        n = min(100, len(df_list) // 50 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(rolling_90_t_test_p_value, df_list, chunksize=10)
    return df_list


def rolling_120_t_test_p_value_parallel(df_list, n=None):
    if n is None:
        n = min(100, len(df_list) // 50 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(rolling_120_t_test_p_value, df_list, chunksize=10)
    return df_list


def rolling_240_t_test_p_value_parallel(df_list, n=None):
    if n is None:
        n = min(100, len(df_list) // 50 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(rolling_240_t_test_p_value, df_list, chunksize=10)
    return df_list


def pickle_load_parallel(file_location_list, n=None):
    if n is None:
        n = min(100, len(file_location_list) // 500 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(read_single_pickle, file_location_list, chunksize=10)
    return df_list


def single_factor_pickle_to_df_parallel(pickle_list, n=None):
    if n is None:
        n = min(100, len(pickle_list) // 500 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(single_pickle_to_df, pickle_list, chunksize=10)
    return df_list


def single_factor_pickle_weight_to_dict_parallel(pickle_list, n=None):
    if n is None:
        n = min(100, len(pickle_list) // 500 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        df_list = pool.map(single_pickle_weight_to_dict, pickle_list, chunksize=10)
    return df_list


def delete_file_folder_parallel(folder_list, n=None):
    if n is None:
        n = min(100, len(folder_list) // 500 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        pool.map(delete_file_folder, folder_list, chunksize=10)


def delete_non_use_pickle_parallel(folder_list, n=None):
    if n is None:
        n = min(100, len(folder_list) // 500 + 1)
    with Pool(processes=n, maxtasksperchild=10) as pool:
        # have your pool map the file names to dataframes
        pool.map(delete_non_use_pickle, folder_list, chunksize=10)


class block_result:
    def __init__(self, nv_all, psr, weight_dict, num_seq, weight_columns) -> None:
        self.nv_all = nv_all
        self.psr = psr
        self.weight_dict = weight_dict
        self.num_seq = num_seq
        self.weight_columns = weight_columns

    def psr_threshold_result(self, threshold):
        selected_formula = list(self.psr[self.psr > threshold].index)
        nv_selected = self.nv_all[selected_formula]
        nv_selected_mean = nv_selected.diff().fillna(0).mean(axis=1).cumsum()
        weight_selected_dict = {k: v for k, v in self.weight_dict.items() if k in selected_formula}
        weight_selected_df = pd.DataFrame()
        for i in range(len(selected_formula)):
            formula_one = selected_formula[i]
            if i == 0:
                weight_selected_df = pd.DataFrame(weight_selected_dict[formula_one])
            else:
                weight_selected_df = weight_selected_df + pd.DataFrame(weight_selected_dict[formula_one])
        weight_selected_df = weight_selected_df.div(len(selected_formula))
        weight_selected_df.columns = self.weight_columns
        return nv_selected, weight_selected_dict, nv_selected_mean, weight_selected_df

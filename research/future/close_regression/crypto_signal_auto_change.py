import pandas as pd
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def calculate_signal_df(signal, ret):
    df = pd.concat([pd.DataFrame(signal), pd.DataFrame(ret)], axis=1)
    df.columns = ['signal', 'ret']
    df = df.sort_values('signal')
    df = df.dropna()
    df.index = range(len(df))
    df['cumpnl'] = df['ret'].cumsum()

    return df


def mad(a, para=5):
    m = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - np.nanmedian(a)))
    result = copy.deepcopy(a)
    result[result < m - para * mad] = m - para * mad
    result[result > m + para * mad] = m + para * mad
    return result


def process_signal_df(df, threshold):
    df1 = df.copy()

    df1_selected = pd.concat(
        [df1[df1['cumpnl'] == df1['cumpnl'].min()], df1[df1['cumpnl'] == df1['cumpnl'].max()], df1.iloc[[0, -1], :]])
    df1_selected = df1_selected.sort_values('signal')
    df1_selected['index'] = df1_selected.index
    df1_selected['index_diff'] = df1_selected['index'].diff() / len(df1)
    if df1_selected['index_diff'].iloc[1] < 0.1:
        df1_selected = df1_selected.drop(index=df1_selected.index[1])
        df1_selected['index'] = df1_selected.index
        df1_selected['index_diff'] = df1_selected['index'].diff() / len(df1)

    if df1_selected['index_diff'].iloc[-1] < 0.1:
        df1_selected = df1_selected.drop(index=df1_selected.index[-2])
        df1_selected['index'] = df1_selected.index
        df1_selected['index_diff'] = df1_selected['index'].diff() / len(df1)

    df1_selected['k'] = df1_selected['cumpnl'].diff().div(df1_selected['index'].diff())
    result = pd.DataFrame()
    for i in range(len(df1_selected) - 1):
        if np.abs(df1_selected['k'].iloc[i + 1]) < threshold:
            pass
        else:
            result.loc[i, "low"] = df1_selected['signal'].iloc[i]
            result.loc[i, "upper"] = df1_selected['signal'].iloc[i + 1]
            result.loc[i, "sign"] = np.sign(df1_selected['k'].iloc[i + 1])
    return result


def single_factor_single_asset_signal_graph(signal, ret, save_location=None):
    df = pd.concat([pd.DataFrame(signal), pd.DataFrame(ret)], axis=1)
    df.columns = ['signal', 'ret']
    df = df.sort_values('signal')
    df = df.dropna()
    df.index = range(len(df))
    df['cumpnl'] = df['ret'].cumsum()

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot(df['cumpnl'], color='blue')
    ax[0].set_title("cumpnl")
    ax[0].set_ylabel("cumpnl")
    ax[1].plot(df['signal'], df['cumpnl'], color='darkorange')
    ax[1].set_title("cumpnl v.s. signal")
    ax[1].set_ylabel("cumpnl")
    ax[1].set_xlabel("signal")

    fig.tight_layout()
    if save_location is None:
        plt.show()
    else:
        plt.savefig(save_location)
        plt.close()


# tickers_list_crypto_selected = ['BTC', 'ETH', 'APT', 'ETC', 'SOL', 'DOGE', 'XRP', 'BNB', 'LTC', 'MATIC']

tickers_list_crypto_selected = ['BTC', 'ETH', 'APT', 'SOL', 'DOGE', 'XRP', 'GALA', 'ETC', 'BNB', 'GMT', 'LTC', 'MATIC']
tickers_list_crypto_selected_usdt = [x + 'USDT' for x in tickers_list_crypto_selected]
tickers_list_crypto_selected_busd = [x + 'BUSD' for x in tickers_list_crypto_selected]

close_price_df = pd.read_csv(r"C:\Users\jason.huang\research\close_reg\crypto\close_price.csv", index_col=0)
# para_list = [5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 150, 252, 512, 1024, 2048]
para = 60
signal_linear = pd.read_csv(
    f"C:\\Users\\jason.huang\\research\\close_reg\\crypto\\linear\\close_reg_signal\\ts_mom_roll_{para}.csv",
    index_col=0)
signal_nonlinear = pd.read_csv(
    f"C:\\Users\\jason.huang\\research\\close_reg\\crypto\\nonlinear\\close_reg_signal\\ts_mom_roll_{para}.csv",
    index_col=0)
save_path = f"C:\\Users\\jason.huang\\research\\close_reg\\crypto\\signal_graph\\para_{para}\\"

nv_all_b = pd.DataFrame()
nv_all_c = pd.DataFrame()
tickers_list_crypto_selected_usdt = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT']
for tickers in tickers_list_crypto_selected_usdt:
    # save_location = f"{save_path}{tickers}.jpeg"
    nv = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    nv.columns = ['b', 'c', 'close']
    nv = nv.reset_index()

    data_all = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    data_all.columns = ['b', 'c', 'close']
    data_all['ret'] = data_all['close'].div(data_all['close'].shift(1)).sub(1)
    data_all['target_ret'] = data_all['ret'].shift(-2)
    data_all.index = pd.to_datetime(data_all.index)
    for i in range(len(nv)):
        if i < 4500:
            nv.loc[i, 'signal_b'] = np.nan
            nv.loc[i, 'signal_c'] = np.nan
        else:

            data = data_all.iloc[:i - 1, :]
            data['target_ret_5_mad'] = mad(data['target_ret'], 5)

            df = calculate_signal_df(signal=data['b'], ret=data['target_ret_5_mad'])
            try:
                result1 = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result1 = pd.DataFrame()
            try:
                result2 = process_signal_df(df[df['signal'] < 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result2 = pd.DataFrame()
            result = pd.concat([result1, result2])

            if len(result) == 0:
                nv.loc[i, 'signal_b'] = 0
            else:
                result = result.sort_values('low').reset_index(drop=True)
                r1 = result[result['low'] < nv.loc[i, 'b']][result['upper'] > nv.loc[i, 'b']]
                if len(r1) == 0:
                    nv.loc[i, 'signal_b'] = 0
                else:
                    nv.loc[i, 'signal_b'] = r1['sign'].iloc[0]

            df = calculate_signal_df(signal=data['c'], ret=data['target_ret_5_mad'])
            try:
                result1 = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result1 = pd.DataFrame()
            try:
                result2 = process_signal_df(df[df['signal'] < 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result2 = pd.DataFrame()
            result = pd.concat([result1, result2])

            if len(result) == 0:
                nv.loc[i, 'signal_c'] = 0
            else:
                result = result.sort_values('low').reset_index(drop=True)
                r1 = result[result['low'] < nv.loc[i, 'c']][result['upper'] > nv.loc[i, 'c']]
                if len(r1) == 0:
                    nv.loc[i, 'signal_c'] = 0
                else:
                    nv.loc[i, 'signal_c'] = r1['sign'].iloc[0]
        print(f"{i}/{len(nv)} is ok")

    nv['ret'] = nv['close'].div(nv['close'].shift(1)).sub(1)
    nv['ret_b'] = nv['signal_b'].shift(2).mul(nv['ret'])
    nv['ret_c'] = nv['signal_c'].shift(2).mul(nv['ret'])
    nv['nv_b'] = nv['ret_b'].cumsum()
    nv['nv_c'] = nv['ret_c'].cumsum()

    nv_selected_b = pd.DataFrame(nv['nv_b'])
    nv_selected_b.columns = [tickers]
    nv_all_b = pd.concat([nv_all_b, nv_selected_b], axis=1)

    nv_selected_c = pd.DataFrame(nv['nv_c'])
    nv_selected_c.columns = [tickers]
    nv_all_c = pd.concat([nv_all_c, nv_selected_c], axis=1)

# joint b & c
nv_all_c_b_positive = pd.DataFrame()
nv_all_c_b_negative = pd.DataFrame()
tickers_list_crypto_selected_usdt = [x for x in close_price_df.columns if x[-4:] == 'USDT']
for tickers in tickers_list_crypto_selected_usdt:
    # save_location = f"{save_path}{tickers}.jpeg"
    nv = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    nv.columns = ['b', 'c', 'close']
    nv = nv.reset_index()

    data_all = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    data_all.columns = ['b', 'c', 'close']
    data_all['ret'] = data_all['close'].div(data_all['close'].shift(1)).sub(1)
    data_all['target_ret'] = data_all['ret'].shift(-2)
    data_all.index = pd.to_datetime(data_all.index)
    for i in range(len(nv)):
        if i < 4500:
            nv.loc[i, 'signal_c_b+'] = np.nan
            nv.loc[i, 'signal_c_b-'] = np.nan
        else:
            data = data_all.iloc[:i - 1, :]
            data['target_ret_5_mad'] = mad(data['target_ret'], 5)
            if nv.loc[i, 'b'] > 0 and nv.loc[i, 'c'] < 0:
                nv.loc[i, 'signal_c_b-'] = 0
                df = calculate_signal_df(signal=data[data['b'] > 0]['c'], ret=data[data['b'] > 0]['target_ret_5_mad'])
                # try:
                #     result1 = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
                # except Exception as e:
                #     result1 = pd.DataFrame()
                try:
                    result = process_signal_df(df[df['signal'] < 0], threshold=data['ret'].std() / 100)
                except Exception as e:
                    result = pd.DataFrame()
                if len(result) == 0:
                    nv.loc[i, 'signal_c_b+'] = 0

                else:
                    result = result.sort_values('low').reset_index(drop=True)

                    r1 = result[result['low'] < nv.loc[i, 'b']][result['upper'] > nv.loc[i, 'b']]
                    if len(r1) == 0:
                        nv.loc[i, 'signal_c_b+'] = 0
                    else:
                        nv.loc[i, 'signal_c_b+'] = r1['sign'].iloc[0]
            elif nv.loc[i, 'b'] < 0 and nv.loc[i, 'c'] > 0:
                nv.loc[i, 'signal_c_b+'] = 0
                df = calculate_signal_df(signal=data[data['b'] < 0]['c'], ret=data[data['b'] < 0]['target_ret_5_mad'])
                try:
                    result = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
                except Exception as e:
                    result = pd.DataFrame()

                # result = pd.concat([result1, result2])

                if len(result) == 0:
                    nv.loc[i, 'signal_c_b-'] = 0
                else:
                    result = result.sort_values('low').reset_index(drop=True)
                    r1 = result[result['low'] < nv.loc[i, 'c']][result['upper'] > nv.loc[i, 'c']]
                    if len(r1) == 0:
                        nv.loc[i, 'signal_c_b-'] = 0
                    else:
                        nv.loc[i, 'signal_c_b-'] = r1['sign'].iloc[0]
            else:
                nv.loc[i, 'signal_c_b+'] = 0
                nv.loc[i, 'signal_c_b-'] = 0
        print(f"{tickers} {i}/{len(nv)} is ok")

    nv['ret'] = nv['close'].div(nv['close'].shift(1)).sub(1)
    nv['ret_c_b+'] = nv['signal_c_b+'].shift(2).mul(nv['ret'])
    nv['ret_c_b-'] = nv['signal_c_b-'].shift(2).mul(nv['ret'])
    nv['nv_c_b+'] = nv['ret_c_b+'].cumsum()
    nv['nv_c_b-'] = nv['ret_c_b-'].cumsum()

    nv_selected_b = pd.DataFrame(nv['nv_c_b+'])
    nv_selected_b.columns = [tickers]
    nv_all_c_b_positive = pd.concat([nv_all_c_b_positive, nv_selected_b], axis=1)

    nv_selected_c = pd.DataFrame(nv['nv_c_b-'])
    nv_selected_c.columns = [tickers]
    nv_all_c_b_negative = pd.concat([nv_all_c_b_negative, nv_selected_c], axis=1)

# roll signal
nv_all_b = pd.DataFrame()
nv_all_c = pd.DataFrame()
tickers_list_crypto_selected_usdt = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT']
for tickers in tickers_list_crypto_selected_usdt:
    # save_location = f"{save_path}{tickers}.jpeg"
    nv = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    nv.columns = ['b', 'c', 'close']
    nv = nv.reset_index()

    data_all = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    data_all.columns = ['b', 'c', 'close']
    data_all['ret'] = data_all['close'].div(data_all['close'].shift(1)).sub(1)
    data_all['target_ret'] = data_all['ret'].shift(-2)
    data_all.index = pd.to_datetime(data_all.index)
    for i in range(len(nv)):
        if i < 4500:
            nv.loc[i, 'signal_b'] = np.nan
            nv.loc[i, 'signal_c'] = np.nan
        else:

            data = data_all.iloc[i - 4500:i - 1, :]
            data['target_ret_5_mad'] = mad(data['target_ret'], 5)

            df = calculate_signal_df(signal=data['b'], ret=data['target_ret_5_mad'])
            try:
                result1 = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result1 = pd.DataFrame()
            try:
                result2 = process_signal_df(df[df['signal'] < 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result2 = pd.DataFrame()
            result = pd.concat([result1, result2])

            if len(result) == 0:
                nv.loc[i, 'signal_b'] = 0
            else:
                result = result.sort_values('low').reset_index(drop=True)
                r1 = result[result['low'] < nv.loc[i, 'b']][result['upper'] > nv.loc[i, 'b']]
                if len(r1) == 0:
                    nv.loc[i, 'signal_b'] = 0
                else:
                    nv.loc[i, 'signal_b'] = r1['sign'].iloc[0]

            df = calculate_signal_df(signal=data['c'], ret=data['target_ret_5_mad'])
            try:
                result1 = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result1 = pd.DataFrame()
            try:
                result2 = process_signal_df(df[df['signal'] < 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result2 = pd.DataFrame()
            result = pd.concat([result1, result2])

            if len(result) == 0:
                nv.loc[i, 'signal_c'] = 0
            else:
                result = result.sort_values('low').reset_index(drop=True)
                r1 = result[result['low'] < nv.loc[i, 'c']][result['upper'] > nv.loc[i, 'c']]
                if len(r1) == 0:
                    nv.loc[i, 'signal_c'] = 0
                else:
                    nv.loc[i, 'signal_c'] = r1['sign'].iloc[0]
        print(f"{tickers} {i}/{len(nv)} is ok")

    nv['ret'] = nv['close'].div(nv['close'].shift(1)).sub(1)
    nv['ret_b'] = nv['signal_b'].shift(2).mul(nv['ret'])
    nv['ret_c'] = nv['signal_c'].shift(2).mul(nv['ret'])
    nv['nv_b'] = nv['ret_b'].cumsum()
    nv['nv_c'] = nv['ret_c'].cumsum()

    nv_selected_b = pd.DataFrame(nv['nv_b'])
    nv_selected_b.columns = [tickers]
    nv_all_b = pd.concat([nv_all_b, nv_selected_b], axis=1)

    nv_selected_c = pd.DataFrame(nv['nv_c'])
    nv_selected_c.columns = [tickers]
    nv_all_c = pd.concat([nv_all_c, nv_selected_c], axis=1)

# record signal interval

for tickers in tickers_list_crypto_selected_usdt:
    # save_location = f"{save_path}{tickers}.jpeg"
    nv = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    nv.columns = ['b', 'c', 'close']
    nv = nv.reset_index()

    data_all = pd.concat([signal_linear[tickers], signal_nonlinear[tickers], close_price_df[tickers]], axis=1)
    data_all.columns = ['b', 'c', 'close']
    data_all['ret'] = data_all['close'].div(data_all['close'].shift(1)).sub(1)
    data_all['target_ret'] = data_all['ret'].shift(-2)
    data_all.index = pd.to_datetime(data_all.index)

    result_all = pd.DataFrame()
    for i in range(len(nv)):
        if i < 4500:
            nv.loc[i, 'signal_b'] = np.nan
            nv.loc[i, 'signal_c'] = np.nan
        else:

            data = data_all.iloc[:i - 1, :]
            data['target_ret_5_mad'] = mad(data['target_ret'], 5)

            df = calculate_signal_df(signal=data['b'], ret=data['target_ret_5_mad'])
            try:
                result1 = process_signal_df(df[df['signal'] > 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result1 = pd.DataFrame()
            try:
                result2 = process_signal_df(df[df['signal'] < 0], threshold=data['ret'].std() / 100)
            except Exception as e:
                result2 = pd.DataFrame()
            result = pd.concat([result1, result2])

            if len(result) == 0:
                nv.loc[i, 'signal_b'] = 0
            else:
                result = result.sort_values('low').reset_index(drop=True)
                r1 = result[result['low'] < nv.loc[i, 'b']][result['upper'] > nv.loc[i, 'b']]
                if len(r1) == 0:
                    nv.loc[i, 'signal_b'] = 0
                else:
                    nv.loc[i, 'signal_b'] = r1['sign'].iloc[0]

                result['time'] = nv['datetime'].iloc[i]
                result_all = pd.concat([result_all, result])

        print(f"{i}/{len(nv)} is ok")

    nv['ret'] = nv['close'].div(nv['close'].shift(1)).sub(1)
    nv['ret_b'] = nv['signal_b'].shift(2).mul(nv['ret'])
    nv['ret_c'] = nv['signal_c'].shift(2).mul(nv['ret'])
    nv['nv_b'] = nv['ret_b'].cumsum()
    nv['nv_c'] = nv['ret_c'].cumsum()

    nv_selected_b = pd.DataFrame(nv['nv_b'])
    nv_selected_b.columns = [tickers]
    nv_all_b = pd.concat([nv_all_b, nv_selected_b], axis=1)

    nv_selected_c = pd.DataFrame(nv['nv_c'])
    nv_selected_c.columns = [tickers]
    nv_all_c = pd.concat([nv_all_c, nv_selected_c], axis=1)

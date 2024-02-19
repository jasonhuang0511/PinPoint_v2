import numpy as np
import pandas as pd


def ema(array, param=None):
    if param is None:
        param = 5
    result = np.array(array)
    coef1 = np.exp(-1 / param)
    for i in range(len(array)):
        if i == 0:
            result[i] = array[i]

        else:

            if np.isfinite(array[i]) and not pd.isna(array[i]):
                if not pd.isna(result[i - 1]):
                    result[i] = coef1 * result[i - 1] + (1 - coef1) * array[i]
                else:
                    result[i] = array[i]
            else:
                result[i] = result[i - 1]
    return result


def macd(array, params=None):
    if params is None:
        params = [9, 12, 26]


def bolling_band(array, params=None):
    if params is None:
        params = [5, 10]


def kdj(array):
    pass


def rsi(array):
    pass


def barra_momentum(df, param=None):
    if param is None:
        param = [10, 50]

    def _barra_momentum_array(array, param=param):
        pass

    return df.apply(_barra_momentum_array, param=param)

import numpy as np
import pandas as pd


########################################
#
#   fitness_max_stats_function
#
########################################

def strategy_nv_indicator_annual_ret(ret):
    try:
        if len(ret.dropna()) / len(ret) < 0.8:
            return -1 * np.inf
        else:
            indicator = ret.mean() * 252
            if np.isfinite(indicator):
                return indicator
            else:
                return -1 * np.inf
    except:
        return -1*np.inf


def strategy_nv_indicator_annual_ir(ret):
    try:
        if len(ret.dropna()) / len(ret) < 0.8:
            return -1 * np.inf
        else:
            indicator = ret.mean() / (ret.std() + 0.0001) * 15.8
            if np.isfinite(indicator):
                return indicator
            else:
                return -1 * np.inf
    except:
        return -1*np.inf


def strategy_nv_indicator_annual_calmar(ret):
    try:
        if len(ret.dropna()) / len(ret) < 0.8:
            return -1 * np.inf
        else:
            mdd = (ret.cumsum() - ret.cumsum().cummax()).min()
            indicator = ret.sum() / (mdd + 0.0001)
            if np.isfinite(indicator):
                return indicator
            else:
                return -1 * np.inf
    except:
        return -1*np.inf


def strategy_nv_indicator_annual_ir_plus_ret(ret):
    try:
        if len(ret.dropna()) / len(ret) < 0.8:
            return -1 * np.inf
        else:
            indicator = ret.mean() * 252 * 5 + ret.mean() / (ret.std() + 0.0001) * 15.8
            if np.isfinite(indicator):
                return indicator
            else:
                return -1 * np.inf
    except:
        return -1 * np.inf


def factor_ic_sum_abs(factor, pct):
    try:
        indicator = factor.shift(2).corrwith(pct).sum().abs()
        if np.isfinite(indicator):
            return indicator
        else:
            return -1 * np.inf
    except:
        return -1*np.inf

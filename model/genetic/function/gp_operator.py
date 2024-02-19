import pandas as pd
from model.genetic.GP import DataObj


# from model.genetic.GP import GeneticProgrammingData
# from model.genetic import config


###########################################
#
#
#       dataframe calculation operator
#
#
###########################################
def df_add(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return left.add(right)


def df_sub(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return left.sub(right)


def df_mul(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return left.mul(right)


def df_div(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return left.div(right)


def df_divabs(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return left.div(right.abs())


def df_identy(data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return data


def df_negative(data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return data.mul(-1)


def df_lag_1(data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return data.shift(1)


def df_lag_n(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.shift(n)


def df_diff_lag_1(data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return data.sub(data.shift(1))


###########################################
#
#
#       time series operator
#
#
###########################################
def ts_roll_zscore(df: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return (df.sub(df.rolling(n).mean())).div(df.rolling(n).std())


def ts_roll_sum(df: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return df.rolling(n).sum()


def ts_roll_max(df: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return df.rolling(n).max()


def ts_roll_min(df: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return df.rolling(n).min()


def ts_roll_mean(df: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return df.rolling(n).mean()


def ts_roll_std(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.rolling(n).std()


def ts_roll_skew(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.rolling(n).skew()


def ts_roll_kurt(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.rolling(n).kurt()


def ts_roll_coef_variation(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.rolling(n).std().div(data.rolling(n).mean())


def ts_roll_coef_variation_abs(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.rolling(n).std().div(data.rolling(n).mean().abs())


def ts_roll_bias(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.sub(data.rolling(n).mean())


def ts_roll_bias_pct(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.div(data.rolling(n).mean()).sub(1)


def ts_roll_corr(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return left.rolling(n).corr(right)


def ts_roll_cov(left: pd.DataFrame = pd.DataFrame(), right: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return left.rolling(n).corr(right).mul(left.rolling(n).std()).mul(right.rolling(n).std())


def ts_roll_corr_lag1diff(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return data.rolling(n).corr(data.sub(data.shift(1)))


def ts_roll_bias_div_std(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return (data.sub(data.shift(1))).div(data.rolling(n).std())


def ts_roll_mean_pct(data: pd.DataFrame = pd.DataFrame(), n1: int = 0, n2: int = 0) -> pd.DataFrame:
    ma1 = (data.rolling(n1).mean())
    ma2 = (data.rolling(n2).mean())
    return ma1.div(ma2) - 1


def ts_roll_mean_pct_diff(data: pd.DataFrame = pd.DataFrame(), n1: int = 0, n2: int = 0) -> pd.DataFrame:
    ma1 = (data.rolling(n1).mean()).div(data.shift(n1))
    ma2 = (data.rolling(n2).mean()).div(data.shift(n2))
    return ma1.sub(ma2)


def ts_roll_bias_div_mean(data: pd.DataFrame = pd.DataFrame(), n: int = 0) -> pd.DataFrame:
    return (data.sub(data.shift(1))).div(data.rolling(n).mean())


def ts_roll_acceleration(data: pd.DataFrame = pd.DataFrame(), n1: int = 0, n2: int = 0) -> pd.DataFrame:
    velocity = data.sub(data.shift(n1))
    return (velocity.sub(velocity.shift(n2))).div(abs(velocity).shift(n2))





###########################################
#
#
#       technical indicator pattern
#
#
###########################################


def both(signal1: pd.DataFrame = pd.DataFrame(), signal2: pd.DataFrame = pd.DataFrame(),
         data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal1.add(signal2) >= 2, 0)
    # return data_obj.I.where(signal1.add(signal2) >= 2, 0)


def either(signal1: pd.DataFrame = pd.DataFrame(), signal2: pd.DataFrame = pd.DataFrame(),
           data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal1.add(signal2) >= 1, 0)


def both3(signal1: pd.DataFrame = pd.DataFrame(), signal2: pd.DataFrame = pd.DataFrame(),
          signal3: pd.DataFrame = pd.DataFrame(), data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal1.add(signal2).add(signal3) >= 3, 0)


def both4(signal1: pd.DataFrame = pd.DataFrame(), signal2: pd.DataFrame = pd.DataFrame(),
          signal3: pd.DataFrame = pd.DataFrame(), signal4: pd.DataFrame = pd.DataFrame(),
          data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal1.add(signal2).add(signal3).add(signal4) >= 4, 0)


def both5(signal1: pd.DataFrame = pd.DataFrame(), signal2: pd.DataFrame = pd.DataFrame(),
          signal3: pd.DataFrame = pd.DataFrame(), signal4: pd.DataFrame = pd.DataFrame(),
          signal5: pd.DataFrame = pd.DataFrame(), data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal1.add(signal2).add(signal3).add(signal4).add(signal5) >= 5, 0)


# 收于均线上方
def ti_close_over_mean(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    ma = getattr(data_obj, "close").rolling(n).mean()
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") > ma, 0)
    return sig


# 收于均线下方
def ti_close_below_ma(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    ma = getattr(data_obj, "close").rolling(n).mean()
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") < ma, 0)
    return sig


# 均线上穿
def ti_ma1_over_ma2(n1: int = 1, n2: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    ma1 = getattr(data_obj, "close").rolling(n1).mean()
    ma2 = getattr(data_obj, "close").rolling(n2).mean()
    sig = getattr(data_obj, "I").where(ma1 > ma2, 0)
    return sig


# 长上影线
def ti_long_up_shadow(pct: float = 0.0, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    maxop = getattr(data_obj, "open").where(getattr(data_obj, "open") > getattr(data_obj, "close"),
                                            getattr(data_obj, "close"))
    upshadow = getattr(data_obj, "high").sub(maxop)
    sig = getattr(data_obj, "I").where(upshadow.div(getattr(data_obj, "high").sub(getattr(data_obj, "low"))) > pct, 0)
    return sig


# N天K长上影线
def ti_long_up_shadow_n(pct: float = 0.0, n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    openn = getattr(data_obj, "open").shift(n)
    highn = getattr(data_obj, "high").rolling(n).max()
    lown = getattr(data_obj, "low").rolling(n).min()
    maxop = openn.where(openn > getattr(data_obj, "close"), getattr(data_obj, "close"))
    upshadow = highn.sub(maxop)
    sig = getattr(data_obj, "I").where(upshadow.div(highn.sub(lown)) > pct, 0)
    return sig


# 长下影线
def ti_long_down_shadow(pct: float = 0.0, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    minop = getattr(data_obj, "open").where(getattr(data_obj, "open") < getattr(data_obj, "close"),
                                            getattr(data_obj, "close"))
    downshadow = minop.sub(getattr(data_obj, "low"))
    sig = getattr(data_obj, "I").where(downshadow.div(getattr(data_obj, "high").sub(getattr(data_obj, "low"))) > pct, 0)
    return sig


# N天长下影线
def ti_long_down_shadow_n(pct: float = 0.0, n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    openn = getattr(data_obj, "open").shift(n)
    highn = getattr(data_obj, "high").rolling(n).max()
    lown = getattr(data_obj, "low").rolling(n).min()
    minop = openn.where(openn < getattr(data_obj, "close"), getattr(data_obj, "close"))
    downshadow = minop.sub(lown)
    sig = getattr(data_obj, "I").where(downshadow.div(highn.sub(lown)) > pct, 0)
    return sig


# 短上影线
def ti_short_up_shadow(pct: float = 0.0, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    maxop = getattr(data_obj, "open").where(getattr(data_obj, "open") > getattr(data_obj, "close"),
                                            getattr(data_obj, "close"))
    upshadow = getattr(data_obj, "high").sub(maxop)
    sig = getattr(data_obj, "I").where(upshadow.div(getattr(data_obj, "high").sub(getattr(data_obj, "low"))) < pct, 0)
    return sig


# N天短上影线
def ti_short_up_shadow_n(pct: float = 0.0, n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    openn = getattr(data_obj, "open").shift(n)
    highn = getattr(data_obj, "high").rolling(n).max()
    lown = getattr(data_obj, "low").rolling(n).min()
    maxop = openn.where(openn > getattr(data_obj, "close"), getattr(data_obj, "close"))
    upshadow = highn.sub(maxop)
    sig = getattr(data_obj, "I").where(upshadow.div(highn.sub(lown)) < pct, 0)
    return sig


# 短下影线
def ti_short_down_shadow(pct: float = 0.0, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    minop = getattr(data_obj, "open").where(getattr(data_obj, "open") < getattr(data_obj, "close"),
                                            getattr(data_obj, "close"))
    downshadow = minop.sub(getattr(data_obj, "low"))
    sig = getattr(data_obj, "I").where(downshadow.div(getattr(data_obj, "high").sub(getattr(data_obj, "low"))) < pct, 0)
    return sig


# N天短下影线
def ti_short_down_shadow_n(pct: float = 0.0, n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    openn = getattr(data_obj, "open").shift(n)
    highn = getattr(data_obj, "high").rolling(n).max()
    lown = getattr(data_obj, "low").rolling(n).min()
    minop = openn.where(openn < getattr(data_obj, "close"), getattr(data_obj, "close"))
    downshadow = minop.sub(lown)
    sig = getattr(data_obj, "I").where(downshadow.div(highn.sub(lown)) < pct, 0)
    return sig


# 十字星
def ti_cross_star(pct: float = 0.0, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    tang = abs(getattr(data_obj, "open").sub(getattr(data_obj, "close")))
    alllen = getattr(data_obj, "high").sub(getattr(data_obj, "low"))
    sig = getattr(data_obj, "I").where(tang.div(alllen) < pct, 0)
    return sig


# N天十字星
def ti_cross_star_n(pct: float = 0.0, n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    openn = getattr(data_obj, "open").shift(n)
    highn = getattr(data_obj, "high").rolling(n).max()
    lown = getattr(data_obj, "low").rolling(n).min()
    tang = abs(openn.sub(getattr(data_obj, "close")))
    alllen = highn.sub(lown)
    sig = getattr(data_obj, "I").where(tang.div(alllen) < pct, 0)
    return sig


# 盘中突破N天新高
def ti_high_new_high(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    sig = getattr(data_obj, "I").where(getattr(data_obj, "high") >= getattr(data_obj, "high").rolling(n).max(), 0)
    return sig


# 收盘突破N天新高
def ti_close_new_high(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") >= getattr(data_obj, "close").rolling(n).max(), 0)
    return sig


# 盘中突破N天新低
def ti_low_new_low(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    sig = getattr(data_obj, "I").where(getattr(data_obj, "low") <= getattr(data_obj, "low").rolling(n).min(), 0)
    return sig


# 收盘突破N天新低
def ti_close_new_low(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") <= getattr(data_obj, "close").rolling(n).min(), 0)
    return sig


# 开盘跳空低开低于N天最低
def ti_open_jump_low_n(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histlow = getattr(data_obj, "low").rolling(n).min().shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "open") < histlow, 0)
    return sig


# 开盘跳空低开低于前一天天最低
def ti_open_jump_low(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histlow = getattr(data_obj, "low").shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "open") < histlow, 0)
    return sig


# 收盘调控低开低于N天最低
def ti_close_jump_low_n(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histlow = getattr(data_obj, "low").rolling(n).min().shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") < histlow, 0)
    return sig


# 收盘调控低开低于前一天最低
def ti_close_jump_low(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histlow = getattr(data_obj, "low").shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") < histlow, 0)
    return sig


# 开盘跳空高开
def ti_open_jump_high_n(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histhigh = getattr(data_obj, "high").rolling(n).max().shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "open") > histhigh, 0)
    return sig


# 开盘跳空高开
def ti_open_jump_high(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histhigh = getattr(data_obj, "high").shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "open") > histhigh, 0)
    return sig


# 收盘跳空高开
def ti_close_jump_high_n(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histhigh = getattr(data_obj, "high").rolling(n).max().shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") > histhigh, 0)
    return sig


# 收盘跳空高开
def ti_close_jump_high(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    histhigh = getattr(data_obj, "high").shift(1)
    sig = getattr(data_obj, "I").where(getattr(data_obj, "close") > histhigh, 0)
    return sig


# 开盘价高开
def ti_open_higher(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "open") > getattr(data_obj, "close").shift(1), 0)


# 开盘价低开
def ti_open_lower(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "open") < getattr(data_obj, "close").shift(1), 0)


# 前N天的信号
def ti_lag(signal: pd.DataFrame = pd.DataFrame(), n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return signal.shift(n)


# 反向指标
def ti_neg(signal: pd.DataFrame = pd.DataFrame(), data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").mul(signal.mul(-1).add(1))


# 实体波动大于过去N天均值
def ti_oc_grt_n_mean(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_mean = abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1).rolling(n).mean().shift(1)
    return getattr(data_obj, "I").where(abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1) > n_mean, 0)


# 实体波动大于过去N天最大值
def ti_oc_grt_n_max(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_max = abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1).rolling(n).max().shift(1)
    return getattr(data_obj, "I").where(abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1) > n_max, 0)


# 实体波动小于过去N天最小值
def ti_oc_ls_n_min(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_min = abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1).rolling(n).min().shift(1)
    return getattr(data_obj, "I").where(abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1) < n_min, 0)


# 影线波动大于过去N天均值
def ti_sh_grt_n_mean(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_mean = abs(getattr(data_obj, "high").div(getattr(data_obj, "low")) - 1).rolling(n).mean().shift(1)
    return getattr(data_obj, "I").where(abs(getattr(data_obj, "high").div(getattr(data_obj, "low")) - 1) > n_mean, 0)


# 影线波动大于过去N天最大值
def ti_sh_grt_n_max(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_max = abs(getattr(data_obj, "high").div(getattr(data_obj, "low")) - 1).rolling(n).max().shift(1)
    return getattr(data_obj, "I").where(abs(getattr(data_obj, "high").div(getattr(data_obj, "low")) - 1) > n_max, 0)


# 影线波动小于过去N天最小值
def ti_sh_ls_n_min(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_min = abs(getattr(data_obj, "high").div(getattr(data_obj, "low")) - 1).rolling(n).min().shift(1)
    return getattr(data_obj, "I").where(abs(getattr(data_obj, "high").div(getattr(data_obj, "low")) - 1) < n_min, 0)


# 过去N天出现过信号
def ti_ever_positive(signal: pd.DataFrame = pd.DataFrame(), n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal.rolling(n).sum() > 0, 0)


# 过去N天出现信号大于阈值
def ti_positive_num(signal: pd.DataFrame = pd.DataFrame(), n1: int = 1, n2: int = 1,
                    data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal.rolling(n1).sum() > n2)


# 过去N天没出现过信号
def ti_never_positive(signal: pd.DataFrame = pd.DataFrame(), n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(signal.rolling(n).sum() < 1, 0)


# 收盘是最低价
def ti_close_lowest(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") <= getattr(data_obj, "low"), 0)


# 收盘是最高价
def ti_close_highest(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") >= getattr(data_obj, "high"), 0)


# 开盘是最低价
def ti_open_lowest(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "open") <= getattr(data_obj, "low"), 0)


# 开盘是最高价
def ti_open_highest(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "open") >= getattr(data_obj, "high"), 0)


# 当日上涨
def ti_price_up(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") > getattr(data_obj, "close").shift(1), 0)


# N天上涨
def ti_n_day_up(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") > getattr(data_obj, "close").shift(n), 0)


# 红K线
def ti_red_k(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") > getattr(data_obj, "open"), 0)


# N天K线红
def ti_red_k_n(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") > getattr(data_obj, "open").shift(n), 0)


# 绿K线
def ti_green_k(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") < getattr(data_obj, "open"), 0)


# N天K线绿
def ti_green_k_n(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    return getattr(data_obj, "I").where(getattr(data_obj, "close") < getattr(data_obj, "open").shift(n), 0)


# N天位移路程比超过阈值
def ti_close_movedist_ratio(pct: float = 0.0, n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    move = abs(getattr(data_obj, "close").div(getattr(data_obj, "close").shift(n)) - 1)
    distance = abs(getattr(data_obj, "close").div(getattr(data_obj, "close").shift(1)) - 1).rolling(n).sum()
    return getattr(data_obj, "I").where(move.div(distance) > pct, 0)


# K线实体相对前一日长
def ti_k_longer(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    k_len = abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1)
    return getattr(data_obj, "I").where(k_len > k_len.shift(1), 0)


# K线实体相对前一日短
def ti_k_shorter(data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    k_len = abs(getattr(data_obj, "close").div(getattr(data_obj, "open")) - 1)
    return getattr(data_obj, "I").where(k_len < k_len.shift(1), 0)


# N天K线实体相对前一期长
def ti_n_k_longer(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_k_len = abs(getattr(data_obj, "close").div(getattr(data_obj, "open").shift(n)) - 1)
    return getattr(data_obj, "I").where(n_k_len > n_k_len.shift(n + 1), 0)


# N天K线实体相对前一期短
def ti_n_k_shorter(n: int = 1, data_obj: DataObj = None) -> pd.DataFrame:
    # if data_obj is None:
    #     data_obj = config.data_obj
    n_k_len = abs(getattr(data_obj, "close").div(getattr(data_obj, "open").shift(n)) - 1)
    return getattr(data_obj, "I").where(n_k_len < n_k_len.shift(n + 1), 0)


###########################################
#
#
#       constant integer operator
#
#
###########################################
def int_1() -> int:
    return 1


def int_2() -> int:
    return 2


def int_3() -> int:
    return 3


def int_4() -> int:
    return 4


def int_5() -> int:
    return 5


def int_6() -> int:
    return 6


def int_7() -> int:
    return 7


def int_8() -> int:
    return 8


def int_9() -> int:
    return 9


def int_10() -> int:
    return 10


def int_11() -> int:
    return 11


def int_12() -> int:
    return 12


def int_13() -> int:
    return 13


def int_14() -> int:
    return 14


def int_15() -> int:
    return 15


def int_16() -> int:
    return 16


def int_20() -> int:
    return 20


def int_30() -> int:
    return 30


def int_40() -> int:
    return 40


def int_60() -> int:
    return 60


def int_90() -> int:
    return 90


def int_120() -> int:
    return 120


def int_252() -> int:
    return 252


def int_504() -> int:
    return 504


def int_1000() -> int:
    return 1000


###########################################
#
#
#       constant float operator
#
#
###########################################
def float_001() -> float:
    return 0.01


def float_002() -> float:
    return 0.02


def float_003() -> float:
    return 0.03


def float_004() -> float:
    return 0.04


def float_005() -> float:
    return 0.05


def float_006() -> float:
    return 0.06


def float_007() -> float:
    return 0.07


def float_008() -> float:
    return 0.08


def float_009() -> float:
    return 0.09


def float_01() -> float:
    return 0.1


def float_015() -> float:
    return 0.15


def float_02() -> float:
    return 0.2


def float_025() -> float:
    return 0.25


def float_03() -> float:
    return 0.3


def float_035() -> float:
    return 0.35


def float_04() -> float:
    return 0.4


def float_045() -> float:
    return 0.45


def float_05() -> float:
    return 0.5


def float_06() -> float:
    return 0.6


def float_07() -> float:
    return 0.7


def float_08() -> float:
    return 0.8


def float_09() -> float:
    return 0.9

import pandas as pd
import numpy as np


############################################################
#
#   weighting function
#
############################################################
def weighting_function_simple_long_short(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return factor.applymap(np.sign)


def weighting_function_phi_function(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return factor.applymap(lambda x: x * np.exp(-x * x / 4) / 0.89)


def weighting_function_tanh(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return factor.applymap(np.tanh)


def weighting_function_sigmoid(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return factor.applymap(lambda x: 1 / (1 + np.exp(-1 * x)))


def weighting_function_identity(factor: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    return factor.applymap(lambda x: x)

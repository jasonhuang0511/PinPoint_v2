import pandas as pd
import model.constants.futures as ConstFut

"""
split function set
    return: two pd.DataFrame
        data_train, data_test
"""


def every_other_row(data: pd.DataFrame = pd.DataFrame()) -> (pd.DataFrame, pd.DataFrame):
    data_train = data.iloc[::2, ]
    data_test = data.iloc[1::2, ]
    return data_train, data_test


def ts_split(data: pd.DataFrame = pd.DataFrame(), ratio: float = 0.7) -> (pd.DataFrame, pd.DataFrame):
    index = int(len(data) * ratio)
    data_train = data.iloc[:index, ]
    data_test = data.iloc[index:, ]
    return data_train, data_test


def ts_split_drop_extraordinary_date(data: pd.DataFrame = pd.DataFrame(), ratio: float = 0.7) -> (
        pd.DataFrame, pd.DataFrame):
    import datetime
    delete_dict = ConstFut.daily_strategy_future_delete_date_dict
    remove_date_list = []
    for k in data.columns:
        try:
            remove_date_list.extend(delete_dict[k])
        except:
            pass

    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index.date

    if len(remove_date_list) == 0:
        pass
    else:
        remove_date_list = list(set(remove_date_list))
        remove_date_list = [datetime.date(int(x[:4]), int(x[5:7]), int(x[-2:])) for x in remove_date_list]
        df = df[~df['date'].isin(remove_date_list)]
    # df = df[~df.index.isin(remove_date_list)]
    index = int(len(df) * ratio)
    data_train = df.iloc[:index, :-1]
    data_test = df.iloc[index:, :-1]
    return data_train, data_test


def every_other_day_drop_extraordinary_date(data: pd.DataFrame = pd.DataFrame()) -> (pd.DataFrame, pd.DataFrame):
    import datetime
    delete_dict = ConstFut.daily_strategy_future_delete_date_dict
    remove_date_list = []
    for k in data.columns:
        try:
            remove_date_list.extend(delete_dict[k])
        except:
            pass
    if len(remove_date_list) == 0:
        pass
    else:
        remove_date_list = list(set(remove_date_list))
        remove_date_list = [datetime.date(int(x[:4]), int(x[5:7]), int(x[-2:])) for x in remove_date_list]
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index.date

    if len(remove_date_list) == 0:
        pass
    else:
        remove_date_list = list(set(remove_date_list))
        remove_date_list = [datetime.date(int(x[:4]), int(x[5:7]), int(x[-2:])) for x in remove_date_list]
        df = df[~df['date'].isin(remove_date_list)]

    # every other day
    df1 = pd.DataFrame(df['date'].unique(), columns=['date'])
    df1.index = range(len(df1))
    df1['index'] = [x % 2 for x in range(len(df1))]

    df = pd.merge(left=df, right=df1, on='date', how='inner')

    data_train = df[df['index'] == 0].iloc[:, :-2]
    data_test = df[df['index'] == 1].iloc[:, :-2]
    return data_train, data_test


def every_other_row_drop_extraordinary_date(data: pd.DataFrame = pd.DataFrame()) -> (pd.DataFrame, pd.DataFrame):
    import datetime
    delete_dict = ConstFut.daily_strategy_future_delete_date_dict
    remove_date_list = []
    for k in data.columns:
        try:
            remove_date_list.extend(delete_dict[k])
        except:
            pass
    remove_date_list = list(set(remove_date_list))
    remove_date_list = [datetime.date(int(x[:4]), int(x[5:7]), int(x[-2:])) for x in remove_date_list]
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index.date
    if len(remove_date_list) == 0:
        pass
    else:
        remove_date_list = list(set(remove_date_list))
        remove_date_list = [datetime.date(int(x[:4]), int(x[5:7]), int(x[-2:])) for x in remove_date_list]
        df = df[~df['date'].isin(remove_date_list)]

    data_train = df.iloc[::2, :-1]
    data_test = df.iloc[1::2, :-1]
    return data_train, data_test

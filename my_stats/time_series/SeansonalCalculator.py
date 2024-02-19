from contextlib import nullcontext
import datetime
from typing import ItemsView
import pandas as pd
import statsmodels.tsa.seasonal as statsseasonal
from copy import deepcopy
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from dateutil.relativedelta import relativedelta


def add_one(input_number):
    return input_number + 1


def add_seven_days(input_date):
    return input_date + datetime.timedelta(days=7)


def add_one_day(input_date):
    return input_date + datetime.timedelta(days=1)


def add_one_month(input_date):
    converted_date = datetime.datetime.strptime(input_date, '%Y-%m')
    new_date = converted_date + relativedelta(months=1)
    return new_date.strftime('%Y-%m')


# Only contains full year/month -- use input a_ts to control
# Calendar month data/Calendar year data
def average_monthly_shape_for_year(a_ts, shape_function, time_field, output_fields, filter_na):
    min_date_time = min(item[time_field] for item in a_ts)
    max_date_time = max(item[time_field] for item in a_ts)

    current_start_date_time = datetime.datetime(min_date_time.year, min_date_time.month, 1)
    current_year = -1

    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')

    result = []
    while current_start_date_time <= max_date_time:
        current_month = current_start_date_time.month
        # Calculate year average as base
        if current_start_date_time.year > current_year:
            current_year = current_start_date_time.year
            current_year_data = list(filter(lambda data_item: data_item[time_field].year == current_year, a_ts))
            # Filter out items if any field in output_fields is null
            for output_field in output_fields:
                current_year_data = list(filter(lambda data_item: data_item[output_field], current_year_data))

            if (current_year_data and len(current_year_data) > 0):
                average_year_data = {
                    'year': current_year
                }
                for output_field in output_fields:
                    average_year_data[output_field] = sum([item[output_field] for item in current_year_data]) / len(
                        current_year_data)

        current_month_data = list(filter(lambda data_item: data_item[time_field].month == current_month
                                                           and data_item[time_field].year == current_year, a_ts))

        # Filter out items if any field in output_fields is null
        for output_field in output_fields:
            current_month_data = list(filter(lambda data_item: data_item[output_field], current_month_data))

        if (current_month_data and len(current_month_data) > 0):
            average_month_data = {
                'year': current_year,
                'month': current_month
            }
            for output_field in output_fields:
                average_month_data[output_field] = sum([item[output_field] for item in current_month_data]) / len(
                    current_month_data)
                output_ratio_field = '%sRatio' % output_field
                average_month_data[output_ratio_field] = shape_function(average_month_data[output_field],
                                                                        average_year_data[output_field])
            result.append(average_month_data)

        current_start_date_time = current_start_date_time + relativedelta(months=1)

    df = pd.DataFrame.from_dict(result)
    return df, add_one


# Left close right open
# Which year does the week belong to depends on the week start date
def average_weekly_shape_for_year(a_ts, shape_function, time_field, output_fields, filter_na):
    min_date_time = min(item[time_field] for item in a_ts)
    max_date_time = max(item[time_field] for item in a_ts)

    current_start_date_time = datetime.datetime(min_date_time.year, min_date_time.month, min_date_time.day)
    current_week_day = current_start_date_time.weekday()
    current_start_date_time = current_start_date_time + datetime.timedelta(days=-1 * current_week_day)

    result = []
    current_year = -1
    week = 1

    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')
    while current_start_date_time <= max_date_time:
        # Calculate year average as base
        if current_start_date_time.year > current_year:
            week = 1
            current_year = current_start_date_time.year
            current_year_data = list(filter(lambda data_item: data_item[time_field].year == current_year, a_ts))
            # Filter out items if any field in output_fields is null
            for output_field in output_fields:
                current_year_data = list(filter(lambda data_item: data_item[output_field], current_year_data))

            if (current_year_data and len(current_year_data) > 0):
                average_year_data = {
                    'year': current_year
                }
                for output_field in output_fields:
                    average_year_data[output_field] = sum([item[output_field] for item in current_year_data]) / len(
                        current_year_data)

        current_end_date_time = current_start_date_time + datetime.timedelta(days=7)
        current_week_data = list(filter(lambda data_item: data_item[time_field] >= current_start_date_time
                                                          and data_item[time_field] < current_end_date_time, a_ts))

        # Filter out items if any field in output_fields is null
        for output_field in output_fields:
            current_week_data = list(filter(lambda data_item: data_item[output_field], current_week_data))

        if (current_week_data and len(current_week_data) > 0):
            average_week_data = {
                'year': current_year,
                'week': week,
                'start_date_time': current_start_date_time,
                'end_date_time': current_end_date_time
            }
            for output_field in output_fields:
                average_week_data[output_field] = sum([item[output_field] for item in current_week_data]) / len(
                    current_week_data)
                output_ratio_field = '%sRatio' % output_field
                average_week_data[output_ratio_field] = shape_function(average_week_data[output_field],
                                                                       average_year_data[output_field])
            result.append(average_week_data)

        week = week + 1
        current_start_date_time = current_end_date_time

    df = pd.DataFrame.from_dict(result)
    return df, add_one


def average_daily_shape_for_week(a_ts, shape_function, time_field, output_fields, filter_na):
    min_date_time = min(item[time_field] for item in a_ts)
    max_date_time = max(item[time_field] for item in a_ts)

    current_start_date_time = datetime.datetime(min_date_time.year, min_date_time.month, min_date_time.day)
    current_week_day = current_start_date_time.weekday()
    current_start_date_time = current_start_date_time + datetime.timedelta(days=-1 * current_week_day)

    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')

    result = []
    while current_start_date_time <= max_date_time:
        # Calculate year average as base
        if current_start_date_time.weekday() == 0:
            week_start_date_time = current_start_date_time
            week_end_date_time = week_start_date_time + datetime.timedelta(days=7)
            current_week_data = list(filter(
                lambda data_item: data_item[time_field] >= week_start_date_time and data_item[
                    time_field] < week_end_date_time, a_ts))
            # Filter out items if any field in output_fields is null
            for output_field in output_fields:
                current_week_data = list(filter(lambda data_item: data_item[output_field], current_week_data))

            if (current_week_data and len(current_week_data) > 0):
                average_week_data = {
                    'start_date_time': week_start_date_time,
                    'end_date_time': week_end_date_time
                }

                for output_field in output_fields:
                    average_week_data[output_field] = sum([item[output_field] for item in current_week_data]) / len(
                        current_week_data)

        current_end_date_time = current_start_date_time + datetime.timedelta(days=1)
        current_day_data = list(filter(lambda data_item: data_item[time_field] >= current_start_date_time
                                                         and data_item[time_field] < current_end_date_time, a_ts))

        # Filter out items if any field in output_fields is null
        for output_field in output_fields:
            current_day_data = list(filter(lambda data_item: data_item[output_field], current_day_data))

        if (current_day_data and len(current_day_data) > 0):
            average_day_data = {
                'date_time': current_start_date_time,
                'week_day': current_start_date_time.weekday() + 1
            }
            for output_field in output_fields:
                average_day_data[output_field] = sum([item[output_field] for item in current_day_data]) / len(
                    current_day_data)
                output_ratio_field = '%sRatio' % output_field
                average_day_data[output_ratio_field] = shape_function(average_day_data[output_field],
                                                                      average_week_data[output_field])
            result.append(average_day_data)
        current_start_date_time = current_end_date_time

    df = pd.DataFrame.from_dict(result)
    return df, add_seven_days


def average_hourly_shape_for_day(a_ts, shape_function, time_field, output_fields, filter_na):
    min_date_time = min(item[time_field] for item in a_ts)
    max_date_time = max(item[time_field] for item in a_ts)

    current_start_date_time = datetime.datetime(min_date_time.year, min_date_time.month, min_date_time.day,
                                                min_date_time.hour)
    current_day = datetime.datetime.min.date()

    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')

    result = []
    while current_start_date_time <= max_date_time:
        if (current_start_date_time.date() > current_day):
            current_day = current_start_date_time.date()
            current_day_data = list(filter(lambda data_item: data_item[time_field].date() == current_day, a_ts))
            for output_field in output_fields:
                current_day_data = list(filter(lambda data_item: data_item[output_field], current_day_data))

            if (current_day_data and len(current_day_data) > 0):
                average_day_data = {
                    'date': current_day
                }
                for output_field in output_fields:
                    average_day_data[output_field] = sum([item[output_field] for item in current_day_data]) / len(
                        current_day_data)

        current_end_date_time = current_start_date_time + datetime.timedelta(hours=1)
        current_hour_data = list(filter(
            lambda data_item: data_item[time_field] >= current_start_date_time and data_item[
                time_field] < current_end_date_time, a_ts))

        # Filter out items if any field in output_fields is null
        for output_field in output_fields:
            current_hour_data = list(filter(lambda data_item: data_item[output_field], current_hour_data))

        if (current_hour_data and len(current_hour_data) > 0):
            average_hour_data = {
                'start_date_time': current_start_date_time,
                'end_date_time': current_end_date_time,
                'hour': current_start_date_time.hour,
                'date': current_day
            }
            for output_field in output_fields:
                average_hour_data[output_field] = sum([item[output_field] for item in current_hour_data]) / len(
                    current_hour_data)
                output_ratio_field = '%sRatio' % output_field
                average_hour_data[output_ratio_field] = shape_function(average_hour_data[output_field],
                                                                       average_day_data[output_field])
            result.append(average_hour_data)
        current_start_date_time = current_end_date_time

    df = pd.DataFrame.from_dict(result)
    return df, add_one_day


def average_shape_days_to_expiry(a_ts, shape_function, group_key_function, current_time_field, expiry_time_field,
                                 output_fields, filter_na, index_grow_function):
    result = []
    expiry_group = []
    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')

    for output_field in output_fields:
        a_ts = list(filter(lambda data_item: data_item[output_field], a_ts))

    if (a_ts and len(a_ts) > 0):
        base_value = {}
        for output_field in output_fields:
            base_value[output_field] = sum([item[output_field] for item in a_ts]) / len(a_ts)

        expiry_group = []
        for data_item in a_ts:
            days_to_expiry = (data_item[expiry_time_field].date() - data_item[current_time_field].date()).days
            group_key = group_key_function(data_item[current_time_field])
            expiry_group_item = next((item for item in expiry_group if
                                      item['group_key'] == group_key and item['days_to_expiry'] == days_to_expiry),
                                     None)
            if (expiry_group_item):
                expiry_group_item['data'].append(data_item)
            else:
                expiry_group_item = {
                    'group_key': group_key,
                    'days_to_expiry': days_to_expiry,
                    'data': [data_item]
                }
                expiry_group.append(expiry_group_item)

        for group_item in expiry_group:
            base_field = min(item[current_time_field] for item in group_item['data']).strftime('%Y-%m')
            result_item = {
                'group_key': group_item['group_key'],
                'days_to_expiry': group_item['days_to_expiry'],
                'base_field': base_field
            }
            for output_field in output_fields:
                result_item[output_field] = sum([item[output_field] for item in group_item['data']]) / len(
                    group_item['data'])
                output_ratio_field = '%sRatio' % output_field
                result_item[output_ratio_field] = shape_function(result_item[output_field], base_value[output_field])
            result.append(result_item)

    df = pd.DataFrame.from_dict(result)
    return df, index_grow_function


# period means seasonal period, if we pass a df with date time index, statsseasonal will get get the seasonal period automatically
# Eg. period will be 12 if we pass in monthly data
def seasonal_decompose(a_ts, group_key_function, time_field, output_fields, filter_na, model='additive', period=None,
                       seasonal=7):
    if model != 'additive' and model != 'multiplicative' and model != 'LOESS':
        return None
    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')

    for output_field in output_fields:
        a_ts = list(filter(lambda data_item: data_item[output_field], a_ts))

    formatted_data = {}
    for a_ts_item in a_ts:
        formatted_data[a_ts_item[time_field]] = {}
        for output_field in output_fields:
            formatted_data[a_ts_item[time_field]][output_field] = a_ts_item[output_field]

    df = pd.DataFrame.from_dict(formatted_data, orient='index')
    for output_field in output_fields:
        if model == 'additive' or model == 'multiplicative':
            # optional parameter
            seasonal_result = statsseasonal.seasonal_decompose(df[output_field], model=model, period=period)
        elif model == 'LOESS':
            fit_data = pd.Series(df[output_field], index=df.index)
            # seasonal parameter is 7 by default
            stl = STL(fit_data, period=period, seasonal=seasonal)
            seasonal_result = stl.fit()
        seasonal_column = '%sSeasonal' % output_field
        df[seasonal_column] = seasonal_result.seasonal
    file_name = 'SeasonalDecompose.csv'
    df.to_csv('OutputData/RatioCalculation/%s' % file_name, index=True, header=True)

    second_result = []
    df['group_key'] = [group_key_function(index) for index, row in df.iterrows()]
    grouped_df = df.groupby('group_key')
    for name, group in grouped_df:
        group_data_list = group.to_dict('records')
        second_result_item = {'group_key': name}
        for output_field in output_fields:
            output_ratio_field = '%sSeasonal' % output_field
            second_result_item[output_ratio_field] = sum([item[output_ratio_field] for item in group_data_list]) / len(
                group_data_list)

        second_result.append(second_result_item)

    second_df = pd.DataFrame.from_dict(second_result)
    file_name = 'SeasonalDecomposeSecond.csv'
    second_df.to_csv('OutputData/RatioCalculation/%s' % file_name, index=False, header=True)

    return df


# period means seasonal period, if we pass a df with date time index, statsseasonal will get get the seasonal period automatically
# average period and min obs must be equal or greater than 2
# Eg. period will be 12 if we pass in monthly data
def stats_decompose(a_ts, is_rolling, average_period, min_obs, base_field, time_field, group_key_function,
                    index_grow_function, output_fields, filter_na, model='additive', period=None, seasonal=7):
    if model != 'additive' and model != 'multiplicative' and model != 'LOESS':
        return None
    if (not filter_na):
        a_ts_data_frame = pd.DataFrame.from_dict(a_ts)
        a_ts_data_frame = a_ts_data_frame.fillna(method="ffill")
        a_ts = a_ts_data_frame.to_dict('records')

    for output_field in output_fields:
        a_ts = list(filter(lambda data_item: data_item[output_field], a_ts))

    df = pd.DataFrame.from_dict(a_ts)
    grouped_df = df.groupby(base_field, as_index=False)

    base_list = []
    for name, group in grouped_df:
        base_list.append(name)

    base_list.sort()
    current_index = 0
    result = []
    if not is_rolling:
        average_period = len(base_list)
    while (current_index <= len(base_list)):
        if (average_period and current_index >= average_period) or (min_obs and current_index >= min_obs):
            if average_period and current_index >= average_period:
                temp_base_list = base_list[current_index - average_period: current_index]
            else:
                temp_base_list = base_list[0: current_index]
            current_base = base_list[current_index - 1]
            current_base = index_grow_function(current_base)
            temp_data = []
            for base_item in temp_base_list:
                sorted_group = grouped_df.get_group(base_item).to_dict('records')
                sorted_group.sort(key=lambda item: item[time_field])
                temp_data.extend(sorted_group)

            formatted_data = {}
            for a_ts_item in temp_data:
                formatted_data[a_ts_item[time_field]] = {}
                formatted_data[a_ts_item[time_field]][time_field] = a_ts_item[time_field]
                formatted_data[a_ts_item[time_field]][base_field] = a_ts_item[base_field]
                for output_field in output_fields:
                    formatted_data[a_ts_item[time_field]][output_field] = a_ts_item[output_field]

            df = pd.DataFrame.from_dict(formatted_data, orient='index')
            for output_field in output_fields:
                if model == 'additive' or model == 'multiplicative':
                    # optional parameter
                    seasonal_result = statsseasonal.seasonal_decompose(df[output_field], model=model, period=period)
                elif model == 'LOESS':
                    fit_data = pd.Series(df[output_field], index=df.index)
                    # seasonal parameter is 7 by default
                    stl = STL(fit_data, period=period, seasonal=seasonal)
                    seasonal_result = stl.fit()
                seasonal_column = '%sSeasonal' % output_field
                df[seasonal_column] = seasonal_result.seasonal

            df['group_key'] = [group_key_function(index) for index, row in df.iterrows()]
            second_grouped_df = df.groupby('group_key')
            for name, group in second_grouped_df:
                group_data_list = group.to_dict('records')
                second_result_item = {
                    base_field: current_base,
                    'group_key': name
                }
                for output_field in output_fields:
                    output_ratio_field = '%sSeasonal' % output_field
                    second_result_item[output_ratio_field] = sum(
                        [item[output_ratio_field] for item in group_data_list]) / len(group_data_list)

                result.append(second_result_item)
        else:
            base_item = base_list[current_index]
            sorted_group = grouped_df.get_group(base_item).to_dict('records')
            sorted_group.sort(key=lambda item: item[time_field])
            for sorted_group_item in sorted_group:
                result_item = {
                    base_field: base_item,
                    'group_key': group_key_function(sorted_group_item[time_field])
                }

                result.append(result_item)

        current_index = current_index + 1

    result_df = pd.DataFrame.from_dict(result)
    grouped_df = result_df.groupby('group_key')
    grouped_result = []
    for name, group in grouped_df:
        group_data_list = group.to_dict('records')
        grouped_result.extend(group_data_list)

    grouped_result_df = pd.DataFrame.from_dict(grouped_result)
    file_name = 'RatioSecond.csv'
    grouped_result_df.to_csv('OutputData/RatioCalculation/%s' % file_name, index=False, header=True)
    return result_df


def weekday_group(input_date_time):
    return datetime.datetime.strptime(input_date_time, '%Y-%m-%d').weekday()


def month_group(input_date_time):
    return '%s' % (input_date_time.month)


# How to use the last entry? handle it innner function
def average_decompose(calculate_function, output_fields, group_filed, base_field, average_period, is_rolling, min_obs,
                      *args, **kwargs):
    df, index_grow_function = calculate_function(*args, **kwargs)
    if is_rolling:
        result = []
        grouped_df = df.groupby(group_filed, as_index=False)
        for name, group in grouped_df:
            sorted_group = group.to_dict('records')
            sorted_group.sort(key=lambda item: item[base_field])
            sorted_group.append({base_field: index_grow_function(sorted_group[-1][base_field])})
            for index, row in enumerate(sorted_group):
                alternative_data = None
                # average([index - average_period : index - 1])
                if (average_period):
                    if (index >= average_period):
                        alternative_data = sorted_group[index - average_period: index]
                # average([0 : index - 1]) if index >= min_obs
                elif (min_obs and index >= min_obs):
                    alternative_data = sorted_group[0: index]
                # average([0 : index - 1]) if not min_obs and index >= 1
                elif (not min_obs and index >= 1):
                    alternative_data = sorted_group[0: index]
                result_item = {
                    base_field: row[base_field],
                    group_filed: name
                }
                if (alternative_data != None and len(alternative_data) > 0):
                    for output_field in output_fields:
                        output_ratio_field = '%sRatio' % output_field
                        result_item[output_ratio_field] = sum(
                            list(map(lambda item: item[output_ratio_field], alternative_data))) / len(alternative_data)
                result.append(result_item)
    else:
        group_result = []
        grouped_df = df.groupby(group_filed)
        for name, group in grouped_df:
            group_data_list = group.to_dict('records')
            group_result_item = {group_filed: name}
            for output_field in output_fields:
                output_ratio_field = '%sRatio' % output_field
                group_result_item[output_ratio_field] = sum(
                    [item[output_ratio_field] for item in group_data_list]) / len(group_data_list)
            group_result.append(group_result_item)

        result = []
        for index, row in df.iterrows():
            result_item = deepcopy(
                next((item for item in group_result if item[group_filed] == row[group_filed]), "none"))
            result_item[base_field] = row[base_field]
            result.append(result_item)

    result_df = pd.DataFrame.from_dict(result)
    file_name = 'RatioSecond.csv'
    result_df.to_csv('OutputData/RatioCalculation/%s' % file_name, index=False, header=True)
    return result_df


if __name__ == '__main__':
    co2 = [
        315.58,
        316.39,
        316.79,
        317.82,
        318.39,
        318.22,
        316.68,
        315.01,
        314.02,
        313.55,
        315.02,
        315.75,
        316.52,
        317.10,
        317.79,
        319.22
    ]

    co2 = pd.Series(
        co2, index=pd.date_range("1-1-1959", periods=len(co2), freq="M"), name="CO2"
    )

    stl = STL(co2, seasonal=13)
    res = stl.fit()
    print('Done')

    df = pd.read_csv(r'InputData/MonthToYearRatio200201-200412.csv', header=0)
    df_dict = df.to_dict('records')
    fields = ['open', 'high', 'low', 'close', 'vol']
    input_data = []
    for df_dict_item in df_dict:
        input_data_item = {
            'date_time': datetime.datetime.strptime('%s-%s' % (df_dict_item['year'], df_dict_item['month']), '%Y-%m')}
        for field in fields:
            input_data_item[field] = df_dict_item[field]
        input_data.append(input_data_item)

    data_frame_seasonal = seasonal_decompose(input_data, month_group, 'date_time', fields, True, model='LOESS')

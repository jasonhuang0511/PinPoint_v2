from tqsdk import TqApi, TqAuth
import datetime
import data.ConstantData.future_basic_information as ConstFut
import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

api = TqApi(auth=TqAuth('jasonhuang0801', 'Asdf123409-'))


def transfer_windcode_to_tq_code(windcode):
    product_str = windcode.upper().split('.')[0][:-4]
    date_str = windcode.upper().split('.')[0][-4:]
    exchange_str = windcode.upper().split('.')[1]

    exchange_str_tq = ConstFut.exchange_code_mapping_from_windcode_to_tianqin[exchange_str]
    if exchange_str == 'SHF' or exchange_str == 'DCE':
        tq_code = f"{exchange_str_tq}.{product_str.lower()}{date_str}"
    elif exchange_str == 'CZC':
        tq_code = f"{exchange_str_tq}.{product_str}{date_str[1:]}"
    else:
        tq_code = f"{exchange_str_tq}.{product_str}{date_str}"

    return tq_code


def get_tickers_latest_price(tickers, time_str='13:30:00.000000'):
    threshold_datetime = f"{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')} {time_str}"
    tq_code = transfer_windcode_to_tq_code(windcode=tickers)
    flag = True
    quote = api.get_quote(tq_code)
    price = quote.last_price
    while flag:
        api.wait_update()
        if quote.datetime > threshold_datetime:
            flag = False
        else:
            price = quote.last_price
            print(f"quote time{quote.datetime}, price {price}")
    return price


def get_open_price(tickers):
    tq_code = transfer_windcode_to_tq_code(windcode=tickers)
    klines = api.get_kline_serial(symbol=tq_code, duration_seconds=60 * 60 * 24, data_length=1)
    open_price = klines['open'][0]
    return open_price


def get_open_15_price(tickers):
    tq_code = transfer_windcode_to_tq_code(windcode=tickers)
    klines = api.get_kline_serial(symbol=tq_code, duration_seconds=60 * 15, data_length=10)
    close_price = klines['close'][0]
    return close_price


def get_tickers_signal1(tickers, basis):
    open_price = get_open_price(tickers)
    morning_price = get_open_15_price(tickers)
    return (morning_price - open_price) / basis, open_price, morning_price


def get_tickers_signal2(tickers, time_str, basis):
    open_price = get_open_price(tickers)
    afternoon_price = get_tickers_latest_price(tickers=tickers, time_str=time_str)
    return (afternoon_price - open_price) / basis, open_price, afternoon_price


def create_trading_signal(tickers, threshold_long=25, threshold_short=-25, signal_1=None, time_str=None):
    sql_string = f"select current_main_instrumentid from future.t_oi_main_contract_map_daily where main_contract_code=\'{tickers}\' order by tradingday DESC limit 1"
    main_contract_windcode = ExtractDataPostgre.sql_query_from_qa(sql_statement=sql_string)
    main_contract_windcode = main_contract_windcode.iloc[0, 0]

    signal1, open1, morning1 = get_tickers_signal1(tickers=main_contract_windcode, basis=0.005)


    if time_str is None:
        time_str = '13:30:00.000000'
    signal2, open2, afternoon2 = get_tickers_signal2(tickers=main_contract_windcode, basis=0.005, time_str=time_str)

    if signal1 > 0 and signal2 >= 0 and signal1 + signal2 > threshold_long:
        signal = 1
    elif signal1 < 0 and signal2 <= 0 and signal1 + signal2 < threshold_short:
        signal = -1
    else:
        signal = 0
    print(f"signal1:{signal1} {open1} {morning1} {(morning1 - open1) / 0.005}")

    print(f"signal2:{signal2} {open2} {afternoon2} {(morning1 - open1) / 0.005}")
    return signal, main_contract_windcode


def process_future_signal(future_signal, tickers, aum):
    sql_string = f"select current_main_instrumentid from future.t_oi_main_contract_map_daily where main_contract_code=\'{tickers}\' order by tradingday DESC limit 1"
    main_contract_windcode = ExtractDataPostgre.sql_query_from_qa(sql_statement=sql_string)
    main_contract_windcode = main_contract_windcode.iloc[0, 0]

    tq_code = transfer_windcode_to_tq_code(windcode=main_contract_windcode)
    quote = api.get_quote(tq_code)
    price = quote.last_price
    bpv = ConstFut.fut_code_bpv[tickers]
    qty = int(round(aum / price / bpv, 0))
    data = pd.read_csv(
        "C:\\Users\\jason.huang\\research\\scripts_working\\Intraday_Interest_Rate_Future\\TRADE_SAMPLE_TEMPLATE.csv")
    data.loc[0, 'Ticker'] = tickers

    if future_signal == 1:
        data.loc[0, 'Side'] = 'LONG'
        data.loc[0, 'Qty'] = qty
    elif future_signal == -1:
        data.loc[0, 'Side'] = 'SHORT'
        data.loc[0, 'Qty'] = qty
    else:
        data.iloc[0, :] = np.nan
    data.to_csv(
        f"C:\\Users\\jason.huang\\research\\scripts_working\\Intraday_Interest_Rate_Future\\TRADE_SAMPLE_TEMPLATE_{tickers}_20230117.csv")


if __name__ == '__main__':
    tickers = 'T.CFE'
    future_signal, _ = create_trading_signal(tickers=tickers, time_str='13:30:00.000000')
    process_future_signal(future_signal, tickers=tickers, aum=5 * 1000 * 10000 / 2)

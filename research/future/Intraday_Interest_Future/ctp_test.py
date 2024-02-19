import json
import socket
import sys
import urllib.parse
from contextlib import closing
import time
from ctpwrapper import ApiStructure
from ctpwrapper import MdApiPy
import pandas as pd
import datetime
import csv
import os
import numpy as np

import data.ConstantData.future_basic_information as ConstFut
import data.SQL.extract_data_from_postgre as ExtractDataPostgre

config = {
    "investor_id": "209039",
    "broker_id": "9999",
    "password": "PinPoint@test0801",
    "md_server": "tcp://180.168.146.187:10211",
    "trader_server": "tcp://180.168.146.187:10201",
    "app_id": "simnow_client_test",
    "auth_code": "0000000000000000"
}

today_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")

file_path = f"C:\\Users\\jason.huang\\research\\scripts_working\\Intraday_Interest_Rate_Future\\{today_str}\\"
if os.path.exists(file_path):
    try:
        os.mkdir(file_path)
    except:
        try:
            os.makedirs(file_path)
        except Exception as e:
            print(e)
file_location = file_path + 'interest_rate_future_tick_last_price.csv'
template_file_location = "C:\\Users\\jason.huang\\research\\scripts_working\\Intraday_Interest_Rate_Future\\TRADE_SAMPLE_TEMPLATE.csv"
trade_file_location = f"C:\\Users\\jason.huang\\research\\scripts_working\\Intraday_Interest_Rate_Future\\TRADE_SAMPLE_TEMPLATE_{today_str}.csv"


def check_address_port(tcp):
    """
    :param tcp:
    :return:
    """
    host_schema = urllib.parse.urlparse(tcp)

    ip = host_schema.hostname
    port = host_schema.port

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((ip, port)) == 0:
            return True  # OPEN
        else:
            return False  # closed


class Md(MdApiPy):
    """
    """

    def __init__(self, broker_id, investor_id, password, request_id=100, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.login = False
        self.broker_id = broker_id
        self.investor_id = investor_id
        self.password = password
        self._request_id = request_id

    @property
    def request_id(self):
        self._request_id += 1
        return self._request_id

    def OnRspError(self, pRspInfo, nRequestID, bIsLast):
        print("OnRspError:")
        print("requestID:", nRequestID)
        print(pRspInfo)
        print(bIsLast)

    def OnFrontConnected(self):
        """
        :return:
        """
        user_login = ApiStructure.ReqUserLoginField(BrokerID=self.broker_id, UserID=self.investor_id,
                                                    Password=self.password)
        self.ReqUserLogin(user_login, self.request_id)

    def OnFrontDisconnected(self, nReason):
        print("Md OnFrontDisconnected {0}".format(nReason))
        sys.exit()

    def OnHeartBeatWarning(self, nTimeLapse):
        """心跳超时警告。当长时间未收到报文时，该方法被调用。
        @param nTimeLapse 距离上次接收报文的时间
        """
        print('Md OnHeartBeatWarning, time = {0}'.format(nTimeLapse))

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        """
        用户登录应答
        :param pRspUserLogin:
        :param pRspInfo:
        :param nRequestID:
        :param bIsLast:
        :return:
        """
        print("OnRspUserLogin")
        print("requestID:", nRequestID)
        print("RspInfo:", pRspInfo)

        if pRspInfo.ErrorID != 0:
            print("RspInfo:", pRspInfo)
        else:
            print("user login successfully")
            print("RspUserLogin:", pRspUserLogin)
            self.login = True

    def OnRtnDepthMarketData(self, pDepthMarketData):
        """
        行情订阅推送信息
        :param pDepthMarketData:
        :return:
        """
        # print("OnRtnDepthMarketData")
        # print("DepthMarketData:", pDepthMarketData)
        if not os.path.exists(file_location):
            data = pd.DataFrame(columns=['datetime', 'tickers', 'last_price', 'open', 'local_datetime'])
            data.to_csv(file_location, index=False)
        fields = [f"{pDepthMarketData.UpdateTime}.{pDepthMarketData.UpdateMillisec}", pDepthMarketData.InstrumentID,
                  pDepthMarketData.LastPrice, pDepthMarketData.OpenPrice, datetime.datetime.now()]
        print(
            f"updatetime {pDepthMarketData.UpdateTime}.{pDepthMarketData.UpdateMillisec} tickers: {pDepthMarketData.InstrumentID},  last price: {pDepthMarketData.LastPrice}")

        with open(file_location, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        # last_price = str(pDepthMarketData).split(',')[5].split('(')[-1].split(')')[0]
        # tickers = str(pDepthMarketData).split(',')[-4].split('(')[-1].split(')')[0]
        # print(f"{tickers}  {last_price}")
        # return last_price, tickers

    def OnRspSubMarketData(self, pSpecificInstrument, pRspInfo, nRequestID, bIsLast):
        """
        订阅行情应答
        :param pSpecificInstrument:
        :param pRspInfo:
        :param nRequestID:
        :param bIsLast:
        :return:
        """
        print("OnRspSubMarketData")
        print("RequestId:", nRequestID)
        print("isLast:", bIsLast)
        print("pRspInfo:", pRspInfo)
        print("pSpecificInstrument:", pSpecificInstrument)

    def OnRspUnSubMarketData(self, pSpecificInstrument, pRspInfo, nRequestID, bIsLast):
        """
        取消订阅行情应答
        :param pSpecificInstrument:
        :param pRspInfo:
        :param nRequestID:
        :param bIsLast:
        :return:
        """
        print("OnRspUnSubMarketData")
        print("RequestId:", nRequestID)
        print("isLast:", bIsLast)
        print("pRspInfo:", pRspInfo)
        print("pSpecificInstrument:", pSpecificInstrument)


def write_tick_data(tickers_list, duration_seconds=60):
    investor_id = config["investor_id"]
    broker_id = config["broker_id"]
    password = config["password"]
    server = config["md_server"]

    if check_address_port(server):
        print("connect to md sever successfully")
        # 1 create
        # 2 register
        # 3 register front
        # 4 init
        md = Md(broker_id, investor_id, password)
        md.Create()
        md.RegisterFront(server)
        md.Init()

        day = md.GetTradingDay()
        print("trading day:", day)
        print("md login:", md.login)
        if md.login:
            md.SubscribeMarketData(tickers_list)
            time.sleep(duration_seconds)
            # while datetime.datetime.now() < datetime.datetime(2023, 1, 18, 14, 59, 00, 000000):
            #     last_price, tickers = md.SubscribeMarketData(["T2303"])
            #     print(f"{datetime.datetime.now()} main last: {last_price} tickers:{tickers}")
            md.UnSubscribeMarketData(tickers_list)
            # md.Join()
            print("time is ok stop")
    else:
        print("md server is down")


def get_tick_price(tickers='T2303', datetime=None):
    data = pd.read_csv(file_location)
    data = data[data['tickers'] == tickers]
    if datetime is None:
        price = data.iloc[-1, 2]
    else:
        data = data[data['datetime'] <= datetime]
        price = data.iloc[-1, 2]
    return price


def get_open_price(tickers='T2303'):
    data = pd.read_csv(file_location)
    data = data[data['tickers'] == tickers]
    price = data.iloc[0, 3]
    return price


def generate_max_oi_main_contract(tickers):
    sql_string = f"select current_main_instrumentid from future.t_oi_main_contract_map_daily where main_contract_code=\'{tickers}\' order by tradingday DESC limit 1"
    main_contract_windcode = ExtractDataPostgre.sql_query_from_qa(sql_statement=sql_string)
    main_contract_windcode = main_contract_windcode.iloc[0, 0]
    return main_contract_windcode


def process_windcode_to_ctp_code(tickers):
    return tickers.split('.')[0]


def generate_signal(signal1, signal2, long_threshold=25, short_threshold=-25):
    if signal1 > 0 and signal2 >= 0 and signal1 + signal2 > long_threshold:
        signal = 1
    elif signal1 < 0 and signal2 <= 0 and signal1 + signal2 < short_threshold:
        signal = -1
    else:
        signal = 0
    return signal


def generate_signal_list(tickers_list, basis=(0.005, 0.005), long_threshold=25, short_threshold=-25):
    # get main_contract
    main_contract_list = [process_windcode_to_ctp_code(generate_max_oi_main_contract(x)) for x in tickers_list]

    # start_record tick data
    index = 0
    while True:
        if index == 0:
            if datetime.datetime.now().time() >= datetime.time(9, 44, 0, 0):
                write_tick_data(tickers_list=main_contract_list, duration_seconds=60)
                index = 1
            print(f"{datetime.datetime.now().time()} wait")
            time.sleep(1)
        if index == 1:
            if datetime.datetime.now().time() >= datetime.time(13, 29, 0, 0):
                write_tick_data(tickers_list=main_contract_list, duration_seconds=60)
                index = 2
                break
            print(f"{datetime.datetime.now().time()} wait")
            time.sleep(1)

    open_price_list = [get_open_price(x) for x in main_contract_list]
    price1_list = [get_tick_price(x, '09:45:00.0') for x in main_contract_list]
    price2_list = [get_tick_price(x, '13:30:00.0') for x in main_contract_list]
    last_price_list = [get_tick_price(x) for x in main_contract_list]

    signal1 = (np.array(price1_list) - np.array(open_price_list)) / np.array(basis)
    signal2 = (np.array(price2_list) - np.array(open_price_list)) / np.array(basis)
    signal_list = [
        generate_signal(signal1[i], signal2[i], long_threshold=long_threshold, short_threshold=short_threshold) for i in
        range(len(signal1))]
    return signal_list, last_price_list


def process_future_signal(signal_list, last_price_list, tickers_list, aum_list):
    main_contract_list = [generate_max_oi_main_contract(x) for x in tickers_list]
    bpv_list = [ConstFut.fut_code_bpv[x] for x in tickers_list]
    data_all = pd.DataFrame()
    for i in range(len(signal_list)):
        data = pd.read_csv(template_file_location)
        data.loc[0, 'Ticker'] = main_contract_list[i]
        qty = int(round(aum_list[i] / last_price_list[i] / bpv_list[i], 0))
        if signal_list[i] == 1:
            data.loc[0, 'Side'] = 'LONG'
            data.loc[0, 'Qty'] = qty
        elif signal_list[i] == -1:
            data.loc[0, 'Side'] = 'SHORT'
            data.loc[0, 'Qty'] = qty
        else:
            data.iloc[0, :] = np.nan
        data_all = pd.concat([data_all, data])
    data_all = data_all.dropna(how='all')
    data_all.to_csv(trade_file_location, index=False)


if __name__ == "__main__":
    tickers_list = ['T.CFE', 'TF.CFE']
    basis = (0.005, 0.005)
    aum_list = [2500 * 10000, 2500 * 10000]
    signal_list, last_price_list = generate_signal_list(tickers_list=tickers_list, basis=basis, long_threshold=25,
                                                        short_threshold=-25)
    process_future_signal(signal_list, last_price_list, tickers_list, aum_list)

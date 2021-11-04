
from flask import Flask, jsonify, request
from flask_ngrok import run_with_ngrok
import time
import requests
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 250)
import talib
import datetime as dt
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover,barssince,SignalStrategy
from numpy import inf
import threading
from threading import Thread
from pivot import pivot_high,pivot_low
import matplotlib.pyplot as plt



def SMA(values, n):
    return ta.sma(pd.Series(values),length=n)

def EMA(values, n):
    return ta.ema(pd.Series(values),length=n)

def RSI(values, n):
    return ta.rsi(pd.Series(values),length=n)

def LOWEST(values,n):
    return pd.Series(values).rolling(n).min()

def HIGHEST(values,n):
    return pd.Series(values).rolling(n).max()

def STDEV(values,n):
    return talib.STDDEV(pd.Series(values),n)
#just repeat one single value a bunch so u can use it for indicators n stuff
def REPEAT(values,n):
    data = [values] * n
    return pd.Series(data)
#just set the avalible data as an idicator  so u can use it for indicators n stuff
def SETA(values):
    return pd.Series(values).fillna(0)


def f_pr_upper_line_selector(index, x, y1, y2):
    upper_line.loc[index]=[x, y1, y2]

def f_pr_lower_line_selector(index, x, y1, y2):
    lower_line.loc[index]=[x, y1, y2]

def f_pr_min_line_selector(index, x, y1, y2):
    min_line.loc[index]=[x, y1, y2]

def f_pr_mid_line_selector(index, x, y1, y2):
    mid_line.loc[index]=[x, y1, y2]

def f_pr_max_line_selector(index, x, y1, y2):
    max_line.loc[index]=[x, y1, y2]

def SET(values):
    return values

def MOMENTUM(values,n):
    return talib.ROCP(pd.Series(values),n)

def normalise(src):
    hi = src.max()
    lo = src.min()
    return (src[-1]-lo)/(hi-lo)

def NORMALIZE(values,n):
    vals = pd.Series(values)
    last = vals.iloc[-1]
    hi = vals.max()
    lo = vals.min()
    return (last-lo)/(hi-lo)

def MAX(values,n):
    return talib.MAX(pd.Series(values),n)

def MIN(values,n):
    return talib.MIN(pd.Series(values),n)

def SUM(values,n):
    return talib.SUM(pd.Series(values),n)

def STOCH(source,high,low,length):
    stoch = (100 * (source -LOWEST(low,length))/(HIGHEST(high,length) - LOWEST(low,length))).fillna(1)
    return stoch

def valuewhen(condition, source, occurrence):
    return source \
        .reindex(condition[condition].index) \
        .shift(-occurrence) \
        .reindex(source.index) \
        .ffill()   

#needed for the pandas multi threaded proccress
class ThreadWithReturnValue(Thread):
    def init(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.init(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return
# app = Flask(__name__)
# run_with_ngrok(app)
  
# @app.route("/", methods=['POST'])
# def hello():
#     print(request.json)
#     return "Hello Geeks!! from Google Colab"
  
# if __name__ == "__main__":
# #   app.run()
#     app.run()
# import json 
# import requests
# data={
# "message":"AWS TESTER ORDER",# database recording.
# "algo_name":"FA_KC_RSI_BTCUSD_5MIN",# needed to spesify what algo
# "symbol":"BTCUSD",# NO SLASHES ! use bybit symbol list only eg,BTCUSD,BTCUSDT,ETHUSD,XRPUSD,EOSUSD
# "tgt_exposure":0, # 1 = 100% so make sure u scale correctly. 1=nat long , 0 = cash , -1 = 1x short
# "expo_type":"SET_TO",# only SET_TO is allowed
# "max_expo":2.1,# the limit that will safe gard you if tgt expo is too high
# "tgt_price":"60300",#MKT to use the current mkt price or type price manualy 
# "order_side":"buy",# this dose not matter... 
# "order_type":"LIMIT",# by default every trade is limit only for now
# "trade_auth":"True",# If false, it will submit order but will not trade, if true it will have the ability to trade your acct.
# "order_text":"AWS TESTER ORDER!",# Used to store msg within the data base and to telegram. 
# "authToken":"720edc47f442fd579616f13b81162d2460cf7273367d4430d07e9dd0cb2c7945" # needed to auth order text to server.
# }

# headers = {'content-type': 'application/json'}
# json_data = json.dumps(data)
# res = requests.post('https://monkey-trader.com/ALGO_PORT_01',headers=headers, data=json_data,timeout=None)
# print(res)


def f_array_polyreg(_X, _Y):
    _sizeY = _Y.size
    _sizeX = _X.size
    _meanX = np.sum(_X) / _sizeX
    _meanY = np.sum(_Y) / _sizeX
    _meanXY = 0.0
    _meanY2 = 0.0
    _meanX2 = 0.0
    _meanX3 = 0.0
    _meanX4 = 0.0
    _meanX2Y = 0.0

    if _sizeY == _sizeX:
        for _i in range(_sizeY):
            
            _Xi = _X[_i]
            _Yi = _Y[_i]
            
            _meanXY = _meanXY + (_Xi * _Yi)
            _meanY2 = _meanY2 + pow(_Yi, 2)
            _meanX2 = _meanX2 + pow(_Xi, 2)
            _meanX3 = _meanX3 + pow(_Xi, 3)
            _meanX4 = _meanX4 + pow(_Xi, 4)
            _meanX2Y = _meanX2Y + pow(_Xi, 2) * _Yi
        _meanXY = _meanXY / _sizeX
        _meanY2 = _meanY2 / _sizeX
        _meanX2 = _meanX2 / _sizeX
        _meanX3 = _meanX3 / _sizeX
        _meanX4 = _meanX4 / _sizeX
        _meanX2Y = _meanX2Y / _sizeX
    _sXX = _meanX2 - _meanX * _meanX
    _sXY = _meanXY - _meanX * _meanY
    _sXX2 = _meanX3 - _meanX * _meanX2
    _sX2X2 = _meanX4 - _meanX2 * _meanX2
    _sX2Y = _meanX2Y - _meanX2 * _meanY

    _b = (_sXY * _sX2X2 - _sX2Y * _sXX2) / (_sXX * _sX2X2 - _sXX2 * _sXX2)
    _c = (_sX2Y * _sXX - _sXY * _sXX2) / (_sXX * _sX2X2 - _sXX2 * _sXX2)
    _a = _meanY - _b * _meanX - _c * _meanX2


    _predictions = []
    _max_dev = 0.0
    _min_dev = 0.0
    _stdev = 0.0

    for _i in range(_sizeX):
        _Xi = _X[_i]
        _vector = _a + _b * _Xi + _c * _Xi * _Xi
        _predictions.append(_vector)
        _Yi = _Y[_i]
        _diff = _Yi - _vector
        if _diff > _max_dev:
          _max_dev = _diff
        if _diff < _min_dev:
            _min_dev = _diff
        _stdev = _stdev + abs(_diff)
    return [_predictions, _max_dev, _min_dev, _stdev/_sizeX]






def get_mkt_data_csv_bybit(trading_pair,Timeframe):
    ##price_data
    def vwap_hlc3(df):
        q = df.Volume.values
        h = df.High.values
        l = df.Low.values
        c = df.Close.values
        hlc3=((h+l+c)/3)
        return df.assign(vwap_hlc3=(hlc3 * q).cumsum() / q.cumsum()).round(2)
    def vwap_close(df):
        q = df.Volume.values
        p = df.Close.values
        return df.assign(vwap_close=(p * q).cumsum() / q.cumsum()).round(2)
    def vwap_open(df):
        q = df.Volume.values
        o = df.Open.values
        return df.assign(vwap_open=(o * q).cumsum() / q.cumsum()).round(2)
    # data = pd.read_csv('bybit-'+'BTCUSD'+'-30s-data.csv')
    data = pd.read_csv('test.csv')
    # data = pd.read_csv('bybit-'+trading_pair+'-30s-data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index.rename('Date',inplace=True)
    data = data[["open", "close", "high", "low", "volume"]]
    data.ta.adjusted = 'close'
    data = data.groupby(pd.Grouper(freq=Timeframe)).agg({'open':'first',
                                                            'close':'last',
                                                            'high':'max',
                                                            'low':'min',
                                                            'close':'last',
                                                        'volume':'sum'})
    data.columns = ["Open", "Close", "High", "Low", "Volume"]
    data.dropna(inplace=True)
    df_temp=data.copy()
    vwap_hlc3 = df_temp.groupby(df_temp.index.date, group_keys=False).apply(vwap_hlc3)
    vwap_close = df_temp.groupby(df_temp.index.date, group_keys=False).apply(vwap_close)
    vwap_open = df_temp.groupby(df_temp.index.date,group_keys=False).apply(vwap_open)
    #pushing vwap values to main df2 mkt data for backtest
    #df2 will get tampered with , unless you use .copy()
    df3_temp=data.copy()
    df3_temp['vwap_hlc3']=vwap_hlc3['vwap_hlc3']
    df3_temp['vwap_close']=vwap_close['vwap_close']
    # df3_temp['vwap_open']=vwap_open['vwap_open']
    #pushing vwap values to main df2 mkt data for backtest
    data = df3_temp
    return data

def calculate_max(data):
    bt_data = data.df
    # bt_data = data
    bt_data.reset_index(inplace=True)
    price_data = data.Open.s
    # price_data = data.Open

    length = 222
    prices = [data.Open.s]
    # prices = [data.Open]
    indices = bt_data.index.tolist()
    feature_list = ['x1','x2', 'y1upper', 'y2upper', 'y1low', 'y2low', 'y1min', 'y2min', 'y1mid', 'y2mid', 'y1max', 'y2max']
    print(len(bt_data.index.to_numpy()))
    print(len(price_data))
    print('+++++++++++++++++++')
    zero_data = np.zeros(shape=(len(price_data),len(feature_list)))

    blat = pd.DataFrame(zero_data, columns=feature_list)

    high = data.High.s.fillna(0)
    if pivot_high(high,2,2) is not None:
        e = prices.pop()
        i = indices.pop()
        prices.insert(0, data.High.s[-3])
        indices.insert(0,bt_data.index[-3])

    if pivot_low(data.Low.s.fillna(0),2,2) is not None:
        e = prices.pop()
        i = indices.pop()
        prices.insert(0, data.Low.s[-3])
        indices.insert(0, bt_data.index[-3])

    [P, Pmax, Pmin, Pstdev] = f_array_polyreg(bt_data.index.to_numpy() , price_data)

    pr_fractions = 10
    pr_size = len(P)
    pr_step = 1

    # pr_step = round(max(pr_size / pr_fractions, 1))

    # def f_pr_upper_line_selector(index, x, y1, y2):
    #     upper_line.loc[index]=[x, y1, y2]

    # def f_pr_lower_line_selector(index, x, y1, y2):
    #     lower_line.loc[index]=[x, y1, y2]

    # def f_pr_min_line_selector(index, x, y1, y2):
    #     min_line.loc[index]=[x, y1, y2]

    # def f_pr_mid_line_selector(index, x, y1, y2):
    #     mid_line.loc[index]=[x, y1, y2]

    # def f_pr_max_line_selector(index, x, y1, y2):
    #     max_line.loc[index]=[x, y1, y2]

    # feature_list = ['x1', 'x2','y1upper', 'y2upper', 'y1low', 'y2low', 'y1min', 'y2min', 'y1mid', 'y2mid', 'y1max', 'y2max']
    for _i in range(0, (pr_size - pr_step) - 1, pr_step):
        _next_step_index = _i + pr_step
        _line = _i / pr_step
        blat.loc[_i] = [indices[_i], indices[_i + pr_step], P[_i] + Pstdev, P[_i+ pr_step] + Pstdev, P[_i] - Pstdev, P[_i+ pr_step] - Pstdev, P[_i] + Pmin, P[_i+ pr_step] + Pmin, P[_i], P[_i+ pr_step], P[_i]+ Pmax, P[_i+ pr_step]+ Pmax]
    blat = blat.replace(0,pd.np.nan).ffill()
    # print(blat)
    zero_data_1 = np.zeros(shape=(len(price_data),2))
    ymax_feature_list = ['y1max','y1min']
    ymax = pd.DataFrame(zero_data_1, columns=ymax_feature_list)

    print('====================')
    ymax['y1max'] =  blat['y1max']
    ymax['y1min'] = blat['y1min']
    # print(ymax)
    

    return ymax['y1max'], ymax['y1min']


trading_pair = 'BTCUSD'
Timeframe = '30S'
data = get_mkt_data_csv_bybit(trading_pair,Timeframe)
data['Close'] = data['Close'].astype(float)
start_date = '2021-08-06 00:00:00'
end_date = '2022-01-01 00:00:00'
# data = data[(data.index.values>=pd.to_datetime(start_date))&(data.index.values<pd.to_datetime(end_date))]
bt_data = data
data.reset_index(inplace=True)
price_data = data['Close'].to_numpy()
[P, Pmax, Pmin, Pstdev] = f_array_polyreg(data.index.to_numpy() , price_data)


class Poly_Reg(Strategy):
    #enter inputs here

    # for later optimization
    use_tp_sl = 0
    sl_dist = 0.05
    use_tp = 0
    tp_dist = 0.01
    move_to_brake_even = 0.001
    minval = 0.0
    maxval = 0.0
    inLong = False
    inShort = False

    allowLongs = True
    allowShorts = False

    closeNoShort = True
    closeNoLong = False

    useTrailing = False
    poly_reg_param = []

    def init(self):
        self.poly_reg_param = self.I(calculate_max, self.data, name='poly_reg_param', plot=True)
        self.maxval = self.I(SET, self.poly_reg_param[0], name='maxval', plot=True)
        self.minval = self.I(SET, self.poly_reg_param[1], name='minval', plot=True)

    def next(self):

        if (crossover(self.data.Close, self.maxval)):
            # print("crossover")
            # print(self.maxval)
            entry_logic = self.position.size > 0
            exit_logic = self.position.size < 0
            short = self.position.is_short ==False
            if short:
                self.sell()

            if(self.position.is_long == True):
                self.position.close()

        if(crossover(self.minval,self.data.Close)):
            # print("crossunder")
            # print(self.minval)
            entry_logic = self.position.size > 0
            exit_logic = self.position.size < 0
            long = self.position.is_long == False
            if long:
                self.buy()
            if(self.position.is_short == True):
                self.position.close()

bt = Backtest(bt_data, Poly_Reg, commission=.0000,cash=100000,exclusive_orders=True,trade_on_close=False)

stats = bt.run()
bt.plot(resample=False)
print(stats)



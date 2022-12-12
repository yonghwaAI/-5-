from __future__ import annotations
import talib
import numpy as np
import pandas as pd

# from realtime_kiwoom.data_provider import *
# import grpc
# import grpc_python.prediction_pb2 as prediction_pb2
# import grpc_python.prediction_pb2_grpc as prediction_pb2_grpc
from xgboost import XGBClassifier
import pickle


class InputBuilder_BaselineModel:
  #@staticmethod
  def __init__(self, universe: dict):
    self.history_minute_dic = {}
    self.unpack(universe)
    self.build_up_input_features()
    self.__X_test = None
    self.merge_and_make_test_input()

  #@property
  def X_test(self):
    return self.__X_test

  def unpack(self, universe:dict):
    df_069500 = universe['069500']['price_df']   
    df_069500 = pd.DataFrame.from_records([
        {'dt':df_069500.index,'open':df_069500.open, 'high':df_069500.high, 'low':df_069500.low, 'close':df_069500.close, 'volume':df_069500.volume}])
    # 오류 : index는 dt type이 아님
    df_069500['dt'] = pd.to_datetime(df_069500['dt'], format='%Y%m%d%H%M').dt.tz_localize('Asia/Seoul')
    df_069500.set_index('dt')

    df_114800 = universe['114800']['price_df']
    df_114800 = pd.DataFrame.from_records([
        {'dt':df_114800.index,'open':df_114800.open, 'high':df_114800.high, 'low':df_114800.low, 'close':df_114800.close, 'volume':df_114800.volume}])
    df_114800['dt'] = pd.to_datetime(df_114800['dt'], format='%Y%m%d%H%M').dt.tz_localize('Asia/Seoul')
    df_114800.set_index('dt')

    self.history_minute_dic['069500'] = df_069500
    self.history_minute_dic['114800'] = df_114800

  def build_up_input_features(self):
    for _, df in self.history_minute_dic.items():
      self.make_basic_features(df)
    for _, df in self.history_minute_dic.items():
      self.make_window_features(df)
    for _, df in self.history_minute_dic.items():
      self.make_binary_indicators(df)

  def merge_and_make_test_input(self):
    new_cols = ['ma_w', 'ma20_w', 'ma60_w', 'macd_w', 'macdsignal_w', 'macdhist_w', 'rsi_w', 'ad_w', 
            'ts_end', 'ts_start', 'is_higher', 'offset_intra_day']
    compact_minute_dic = {code:df[new_cols] for code, df in self.history_minute_dic.items()}
    merged_df = pd.merge(
      compact_minute_dic['X'], 
      compact_minute_dic['Y'],
      left_index=True, 
      right_index=True, 
      suffixes=('_x', '_y')
      )
    effective_cols = [
       'ma_w_x', 'ma20_w_x', 'ma60_w_x', 'macd_w_x', 'macdsignal_w_x', 'macdhist_w_x', 'rsi_w_x', 'ad_w_x', 'is_higher_x', 
       'offset_intra_day_x', # drop 'ts_end_x', 'ts_start_x'
       'ma_w_y', 'ma20_w_y', 'ma60_w_y', 'macd_w_y', 'macdsignal_w_y', 'macdhist_w_y', 'rsi_w_y', 'ad_w_y', 'is_higher_y'
       ]
    self.__X_test = merged_df[effective_cols].iloc[[-1]]

  def make_basic_features(self, df: pd.DataFrame):
    """
    df가 변형됨
    """
    ma = talib.MA(df['close'], timeperiod=30)
    macd, macdsignal, macdhist = talib.MACD(df['close'])
    rsi = talib.RSI(df['close'], timeperiod=14)
    ad = talib.AD(df['high'], df['low'], df['close'], df['volume'])

    df['ma'] = ma
    df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean()
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist
    df['rsi'] = rsi
    df['ad'] = ad

    df.index = pd.to_datetime(df.index)
    df['offset_intra_day'] = ((df.index - df.index.floor('D') - pd.Timedelta('9h')).total_seconds()/(60*60*6.5)).values
    
  def make_window_features(self, df: pd.DataFrame, cols=['ma', 'macd', 'macdsignal', 'macdhist', 'rsi', 'ad'], window_size=10):
    """
    df가 변형됨: 과거 윈도우 동안의 평균값대비 현재 값의 차이를 계산
    """
    for col in cols:
      prev_summary = df[col].rolling(window=window_size).mean().shift(1)
      df[f'{col}_w'] = (df[col] - prev_summary)  

  def make_binary_dt_features(self, df: pd.DataFrame):
    """
    df가 변형됨
    """
    ss = df.reset_index()
    ss['dt'] = ss['index']
    df['ts_end'] = ss.dt.shift(-1).apply(lambda x: x.hour == 9 and x.minute == 0).values
    df['ts_start'] = ss.dt.apply(lambda x: x.hour == 9 and x.minute == 0).values

  def make_binary_close_indicators(self, df: pd.DataFrame):
    """
    df가 변형됨
    """
    daily_prev_close = df.groupby(df.index.strftime('%Y-%m-%d')).close.last().shift(1)
    xx = pd.Series(df.index.strftime('%Y-%m-%d').map(daily_prev_close).values, index=df.index)
    df['is_higher'] = xx < df.close
    df.loc[xx.isna(), 'is_higher']=np.nan

  def make_binary_indicators(self, df: pd.DataFrame):
    self.make_binary_dt_features(df)
    self.make_binary_close_indicators(df)


class BaselineModel:
  def __init__(self):
    with open('C:/Users/yonghwa/Desktop/4-2/인공지능 금융 통계/AIFT/-AI-financial-investment-team5/model/model_xgboost.pkl', 'rb') as f:
        self.model = pickle.load(f)

  def predict(self, X_test):
    return self.model.predict(X_test)

  def predict_proba(self, X_test):
    return self.model.predict_proba(X_test)
from strategy.RSIStrategy import *
import talib
import pandas as pd
import numpy as np




class tech_model():
    def build_up_input_features(self,df: pd.DataFrame):
        self.make_basic_features(df)
        self.make_window_features(df)
        self.make_binary_indicators(df)
        self.make_target(df,window_size=30)
        

    def make_basic_features(self, df: pd.DataFrame):
        """
        df가 변형됨
        """
        ma = talib.MA(df['close'], timeperiod=30)
        macd, macdsignal, macdhist = talib.MACD(df['close'])
        rsi = talib.RSI(df['close'], timeperiod=14)
        ad = talib.AD(df['high'], df['low'], df['close'], df['volume'])

        df['ma'] = ma
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
      
      
    def make_target(self, df: pd.DataFrame,window_size=30):
        df['target'] = df.close.rolling(window_size).mean().shift(-window_size)/df.close
    
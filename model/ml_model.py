# 모델링 파일... 춥다...

'''
import plotly.figure_factory as ff
import plotly.graph_objects as go
from flaml import AutoML
automl = AutoML()
from sktime.forecasting.model_selection import temporal_train_test_split

from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from flaml.ml import sklearn_metric_loss_score
from flaml.data import get_output_from_log
import matplotlib.pyplot as plt
'''

import pandas as pd
import numpy as np
import talib

class Tech_model():
    def __init__(self):
        super().__init__()
    
    # 1. 기술적 지표 추가
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

    def make_binary_close_indicators(self, df: pd.DataFrame, daily_prev_close_map=None):
        """
        df가 변형됨
        """
        if daily_prev_close_map is None:
            daily_prev_close_map = df.groupby(df.index.strftime('%Y-%m-%d')).close.last().shift(1)
        xx = pd.Series(df.index.strftime('%Y-%m-%d').map(daily_prev_close_map).values, index=df.index)
        df['is_higher'] = xx < df.close
        df.loc[xx.isna(), 'is_higher']=np.nan

    def make_binary_indicators(self, df: pd.DataFrame, daily_prev_close_map=None):
        self.make_binary_dt_features(df)
        self.make_binary_close_indicators(df, daily_prev_close_map)

    def make_target(self, df: pd.DataFrame, window_size=15):
        """
        df가 변형됨
        close의 내일 ~ window_size 까지의 가격 변화율을 target으로 함
        """
        df['target'] = df.close.rolling(window=window_size).mean().shift(-window_size) /df.close

    # 2. 일단위로 모델링
    def get_daily_prev_close_map(self, df: pd.DataFrame):
        """일별 -> 전일자 종가 """
        return df.groupby(df.index.strftime('%Y-%m-%d')).close.last().shift(1)
    
    def get_daily_dic(self, universe: dict):
        dic = {}
        history = {'069500':universe['069500']['price_df'],
                   '114800':universe['114800']['price_df']}

        for code, df in history.items():            
            df.index = pd.to_datetime(df.index)
            print('code:', code, '\n', 'df:\n', df)
            sampler = df.resample('1D')
            daily_prev_close_map = self.get_daily_prev_close_map(df)
            
            datas = []
            for i, (name, group) in enumerate(sampler):
                if len(group) == 0:
                    continue
                daily_df = group.copy()
                self.make_basic_features(daily_df)
                self.make_window_features(daily_df)
                self.make_binary_indicators(daily_df, daily_prev_close_map=daily_prev_close_map)
                self.make_target(daily_df, window_size=30)
                datas.append(daily_df.dropna(axis=0))
            dic[code] = pd.concat(datas)
        print('dic:\n',dic)

        # 코드별 지표컬럼명 변경
        new_cols = ['ma_w', 'macd_w', 'macdsignal_w', 'macdhist_w', 'rsi_w', 'ad_w', 
        'ts_end', 'ts_start', 'is_higher', 'offset_intra_day', 'target']
        compact_minute_dic = {code:df[new_cols] for code, df in dic.items()}
        merged_df = pd.merge(
            compact_minute_dic['069500'], 
            compact_minute_dic['114800'], 
            left_index=True, 
            right_index=True, 
            suffixes=('_x', '_y')
            )

        # merged_df.to_pickle('.merged_for_baseline2_df.pkl')
        merged_df = pd.read_pickle('.merged_for_baseline2_df.pkl')
        print(merged_df)
        print('target_x 분포 확인 : \n', merged_df.target_x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]))

        # Precision이 높으면 threshold가 높아진다. 
        # Recall을 높이면 threshold가 낮아진다.
        # precison을 놓치지 않는 상황에서 recall을 높이는 것이 좋다.
        # ROC 커브의 중간점은 tpr-fpr이 가장 큰 지점으로, 최적의 threshold 값을 가진다.
        
        # 1 - target_x의 상위 95%
        decision_up_threshold= merged_df.target_x.quantile(0.95) - 1
        # 1 - target_x의 하위 5%
        decision_down_threshold= 1 - merged_df.target_x.quantile(0.05)
        print('\nUP: ', decision_up_threshold)
        print('Down : ',decision_down_threshold)
        merged_df['label'] = 'NOP'
        merged_df.loc[(merged_df.target_x > 1 + decision_up_threshold) & (merged_df.target_y < 1 - decision_down_threshold), 'label'] = 'X'
        merged_df.loc[(merged_df.target_x < 1 - decision_down_threshold) & (merged_df.target_y > 1 + decision_up_threshold), 'label'] = 'Y'
        merged_df['label'] = merged_df.label.astype('category')

        merged_df = merged_df.shift(1)

        print(merged_df.label.value_counts(normalize=True))
        print(merged_df.label.value_counts(normalize=False))

        merged_df.to_csv('data.csv', index=True)
        
        # return merged_df
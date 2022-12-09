import talib
import pandas as pd
import numpy as np
import ml_model     



history_provider = MinuteChartDataProvider.Factory(cm, tag='history')

history_minute_dic = history_provider.get_history_from_ndays_ago(n_days=365)

dic = {}
for code, df in history_minute_dic.items():
  sampler = df.resample('1D')
  daily_prev_close_map = get_daily_prev_close_map(df)
  datas = []
  for i, (name, group) in enumerate(sampler):
    if len(group) == 0:
      continue
    daily_df = group.copy()
    make_basic_features(daily_df)
    make_window_features(daily_df)
    make_binary_indicators(daily_df, daily_prev_close_map=daily_prev_close_map)
    make_target(daily_df, window_size=30)
    datas.append(daily_df.dropna(axis=0))
  dic[code] = pd.concat(datas)

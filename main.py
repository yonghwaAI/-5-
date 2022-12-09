from strategy.RSIStrategy import *
import sys

app = QApplication(sys.argv)

# 데이터프레임 모든 컬럼 출력
pd.set_option('display.max_columns', None)

rsi_strategy = RSIStrategy()
rsi_strategy.start()

app.exec_()
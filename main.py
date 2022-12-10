from strategy.Strategy import *
import sys

app = QApplication(sys.argv)

# 데이터프레임 모든 컬럼 출력
pd.set_option('display.max_columns', None)

strategy = Strategy()
strategy.start()

app.exec_()
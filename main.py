from strategy.RSIStrategy import *
import sys

app = QApplication(sys.argv)

pd.set_option('display.max_columns', None)
rsi_strategy = RSIStrategy()
rsi_strategy.start()

app.exec_()



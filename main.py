from strategy.RSIStrategy import *
import sys

app = QApplication(sys.argv)

rsi_strategy = Strategy()

rsi_strategy.start()

app.exec_()

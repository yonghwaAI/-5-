from strategy.RSIStrategy import *
import sys
app = QApplication(sys.argv)

strategy = RSIStrategy()
strategy.start()
print(price_df)
app.exec_()
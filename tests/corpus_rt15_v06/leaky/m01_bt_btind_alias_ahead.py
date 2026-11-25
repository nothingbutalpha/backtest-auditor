# leak: 8
import backtrader as bt
import backtrader.indicators as btind
class MeanRev(bt.Strategy):
    def __init__(self):
        self.rsi = btind.RSI(period=14)
    def next(self):
        if self.rsi[1] < 30 and not self.position:
            self.buy()
        elif self.rsi[0] > 70 and self.position:
            self.close()

# leak: 9
import backtrader as bt
class Macd(bt.Strategy):
    def __init__(self):
        m = bt.ind.MACD(self.data.close)
        self.macd = m.macd
        self.sig = m.signal
    def next(self):
        if self.macd[1] > self.sig[0]:
            self.buy()

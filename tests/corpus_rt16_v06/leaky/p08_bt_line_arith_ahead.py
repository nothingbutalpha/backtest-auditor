# leak: 8
import backtrader as bt
class Spread(bt.Strategy):
    def __init__(self):
        self.spread = self.data0.close - 1.5 * self.data1.close
    def next(self):
        z = self.spread
        if z[2] > 0:
            self.sell(data=self.data0)

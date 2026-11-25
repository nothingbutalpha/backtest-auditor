# expect: none
import backtrader as bt
class Bands(bt.Strategy):
    params = (("bounds", (0.97, 1.03)), ("sizes", [10, 20]))
    def __init__(self):
        self.ma = bt.ind.SMA(self.data.close, period=20)
    def next(self):
        if self.data.close[0] < self.ma[0] * self.p.bounds[0]:
            self.buy(size=self.p.sizes[1])
        elif self.data.close[0] > self.ma[0] * self.params.bounds[1]:
            self.sell(size=self.p.sizes[0])

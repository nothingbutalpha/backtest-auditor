# leak: 9
import backtrader as bt
class Momentum(bt.Strategy):
    def __init__(self):
        self.sma = bt.ind.SMA(self.data.close, period=20)
    def next(self):
        if self.position:
            return
        if self.data.open[1] > self.sma[0]:
            self.buy()
cerebro = bt.Cerebro()
cerebro.addstrategy(Momentum)

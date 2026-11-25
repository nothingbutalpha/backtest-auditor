import backtrader as bt
class St(bt.Strategy):
    def __init__(self):
        self.sma = bt.ind.SMA(period=20)
    def next(self):
        if self.data.close[0] > self.sma[0] and not self.position:
            self.buy()
        elif self.data.close[0] < self.sma[0] and self.position:
            self.close()
cerebro = bt.Cerebro()
cerebro.broker.setcommission(commission=0.001)
cerebro.addstrategy(St)

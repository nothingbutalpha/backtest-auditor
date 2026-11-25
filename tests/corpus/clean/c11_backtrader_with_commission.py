# expect: none
import backtrader as bt


class SmaCross(bt.Strategy):
    params = dict(fast=10, slow=30)

    def __init__(self):
        self.crossover = bt.ind.CrossOver(bt.ind.SMA(period=self.p.fast), bt.ind.SMA(period=self.p.slow))

    def next(self):
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.close()


cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.broker.setcommission(commission=0.001)
cerebro.run()

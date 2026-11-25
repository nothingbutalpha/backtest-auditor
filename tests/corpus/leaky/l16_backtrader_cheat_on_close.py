# expect: LA07:high
import backtrader as bt


class SmaCross(bt.Strategy):
    def __init__(self):
        self.crossover = bt.ind.CrossOver(bt.ind.SMA(period=10), bt.ind.SMA(period=30))

    def next(self):
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.close()


cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.broker.setcommission(commission=0.001)
cerebro.broker.set_coc(True)
cerebro.run()

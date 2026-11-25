# expect: LA13:high@11
import backtrader as bt


class Pullback(bt.Strategy):
    def __init__(self):
        self.sma = bt.ind.SMA(self.data.close, period=30)

    def next(self):
        if not self.position and self.data.close[0] < self.sma[0]:
            if self.data.close[1] > self.data.open[1]:
                self.buy()
        elif self.position and self.data.close[0] > self.sma[0]:
            self.close()


cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.GenericCSVData(dataname="ibm.csv"))
cerebro.addstrategy(Pullback)
cerebro.run()

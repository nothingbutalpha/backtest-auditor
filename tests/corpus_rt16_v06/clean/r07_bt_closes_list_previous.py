# expect: none
import backtrader as bt
import backtrader.indicators as btind
class Pairs(bt.Strategy):
    def __init__(self):
        self.closes = [d.close for d in self.datas]
        self.atr = btind.ATR(self.datas[1], period=14)
    def next(self):
        if self.closes[1][0] > self.closes[1][-1] + self.atr[-1]:
            self.sell(data=self.datas[1])

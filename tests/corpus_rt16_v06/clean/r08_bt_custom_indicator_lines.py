# expect: none
import backtrader as bt
class Mid(bt.Indicator):
    lines = ("mid",)
    params = (("windows", (10, 20)),)
    def __init__(self):
        self.lines.mid = (bt.ind.Highest(self.data.high, period=self.p.windows[1]) + bt.ind.Lowest(self.data.low, period=self.p.windows[1])) / 2
class UseMid(bt.Strategy):
    def __init__(self):
        self.mid = Mid(self.data)
    def next(self):
        if self.data.close[0] > self.mid[-1]:
            self.buy()

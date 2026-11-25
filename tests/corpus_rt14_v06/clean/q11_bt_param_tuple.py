# expect: none
import backtrader as bt
class Cross(bt.Strategy):
    def __init__(self):
        self.periods = (10, 30)
        fast = bt.ind.SMA(self.data.close, period=self.periods[0])
        slow = bt.ind.SMA(self.data.close, period=self.periods[1])
        self.cross = bt.ind.CrossOver(fast, slow)
    def next(self):
        if self.cross[0] > 0:
            self.buy()
        elif self.cross[0] < 0:
            self.close()

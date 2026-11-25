# expect: none
import backtrader as bt
class Rev(bt.Strategy):
    def __init__(self):
        self.levels = [0.98, 1.02]
    def next(self):
        if self.data.close[0] < self.data.close[-1] * self.levels[0]:
            self.buy()
        elif self.data.close[0] > self.data.close[-1] * self.levels[1]:
            self.sell()

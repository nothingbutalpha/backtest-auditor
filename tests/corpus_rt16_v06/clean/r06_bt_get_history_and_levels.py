# expect: none
import backtrader as bt
class Breakout(bt.Strategy):
    def next(self):
        hist = self.data.close.get(size=5)
        levels = (self.data.close[0] * 0.98, self.data.close[0] * 1.02)
        if len(hist) == 5 and hist[1] < levels[1] and self.data.close[0] > max(hist[:-1]):
            self.buy()

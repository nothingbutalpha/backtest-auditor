# leak: 7
import backtrader as bt
class Gap(bt.Strategy):
    def next(self):
        o = self.data.open
        c = self.data.close
        if o[1] > c[0] * 1.01:
            self.buy()

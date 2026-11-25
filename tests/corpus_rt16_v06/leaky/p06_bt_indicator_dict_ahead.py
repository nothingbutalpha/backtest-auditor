# leak: 8
import backtrader as bt
class Multi(bt.Strategy):
    def __init__(self):
        self.smas = {d: bt.ind.SMA(d.close, period=50) for d in self.datas}
    def next(self):
        for d in self.datas:
            if d.close[0] > self.smas[d][1]:
                self.order_target_percent(d, 0.1)

# expect: none
import backtrader as bt
class Rotate(bt.Strategy):
    def __init__(self):
        self.orders = {}
        self.ranked = sorted(self.datas, key=lambda d: d._name)
        self.moms = {d: bt.ind.ROC(d.close, period=60) for d in self.datas}
    def next(self):
        best = max(self.datas, key=lambda d: self.moms[d][0])
        second = sorted(self.datas, key=lambda d: self.moms[d][0])[-2]
        if self.ranked[1] is best or second is best:
            self.order_target_percent(best, 0.5)

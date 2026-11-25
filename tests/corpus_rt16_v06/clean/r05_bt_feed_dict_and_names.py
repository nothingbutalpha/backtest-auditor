# expect: none
import backtrader as bt
class Rank(bt.Strategy):
    def __init__(self):
        self.by_name = {d._name: d for d in self.datas}
        self.names = sorted(self.by_name)
        self.rocs = {d: bt.ind.ROC(d.close, period=20) for d in self.datas}
    def next(self):
        second = self.by_name[self.names[1]]
        if self.rocs[second][0] > self.rocs[self.datas[0]][0]:
            self.order_target_percent(second, 0.5)

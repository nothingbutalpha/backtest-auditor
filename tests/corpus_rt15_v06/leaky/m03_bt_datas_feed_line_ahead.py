# leak: 6
import backtrader as bt
class Pair(bt.Strategy):
    def next(self):
        a = self.datas[0]
        if self.datas[1].close[1] > self.datas[1].close[0]:
            self.buy(data=a)

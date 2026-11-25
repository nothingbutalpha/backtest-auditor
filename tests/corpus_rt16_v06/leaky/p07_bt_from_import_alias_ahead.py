# leak: 9
import backtrader as bt
from backtrader.indicators import BollingerBands as BB
class Band(bt.Strategy):
    def __init__(self):
        self.bb = BB(self.data, period=20)
    def next(self):
        top = self.bb.lines.top
        if self.data.close[0] < top[1]:
            self.sell()

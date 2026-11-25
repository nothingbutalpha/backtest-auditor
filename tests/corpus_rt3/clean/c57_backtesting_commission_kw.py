# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
class Sma(Strategy):
    def init(self):
        self.a = self.I(lambda x: x, self.data.Close)
    def next(self):
        pass
bt = Backtest(df, Sma, cash=10_000, **{"commission": 0.002})
print(bt.run())

# expect: CO01:possible
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


class Cross(Strategy):
    def init(self):
        self.fast = self.I(SMA, self.data.Close, 10)
        self.slow = self.I(SMA, self.data.Close, 30)

    def next(self):
        if crossover(self.fast, self.slow):
            self.buy()
        elif crossover(self.slow, self.fast):
            self.position.close()


stats = Backtest(GOOG, Cross, cash=10_000).run()
print(stats["Sharpe Ratio"])

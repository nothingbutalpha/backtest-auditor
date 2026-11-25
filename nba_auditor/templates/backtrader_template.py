"""Backtest Auditor template: backtrader (one asset, daily bars, long or flat).

Timing conventions (keep them when you change the rule):
  1. next() decides from the current bar and earlier ones only: index [0] is the bar that just closed,
     [-1] the one before. Never read [1] or later.
  2. Market orders fill at the next bar's open, backtrader's default. Don't enable cheat-on-close.
  3. Commission and slippage are parameters. Never set them to zero by default.
  4. Train and test are split by date. Choose parameters on the train period only, and count every
     variant you tried (rules, parameters, assets) as n_trials when you audit the results.

Input: a CSV you supply with columns date, open, high, low, close, volume (split- and dividend-adjusted,
all the same way). Output: net daily returns in returns.csv, ready for the auditor's audit_returns or
audit_backtest. Results are hypothetical, simulated past performance, not a prediction or guarantee.
"""
import backtrader as bt
import pandas as pd

DATA_CSV = "prices.csv"      # your data: date, open, high, low, close, volume
TRAIN_END = "2018-12-31"     # last date of the train period; everything after it is the test period
FAST, SLOW = 20, 100         # example parameters: replace with your own rule's
COMMISSION = 0.0005          # commission per trade as a fraction of the trade value (0.0005 = 5 bps)
SLIPPAGE = 0.0005            # slippage per trade as a fraction of the price (0.0005 = 5 bps)
CASH = 100_000.0


class Strategy(bt.Strategy):
    """Example rule, to be replaced by yours: long while the fast moving average is above the slow one."""

    params = (("fast", FAST), ("slow", SLOW))

    def __init__(self):
        self.fast = bt.indicators.SMA(self.data.close, period=self.p.fast)  # trailing averages
        self.slow = bt.indicators.SMA(self.data.close, period=self.p.slow)

    def next(self):
        want_long = self.fast[0] > self.slow[0]      # decided at this bar's close
        if want_long and not self.position:
            self.order_target_percent(target=0.95)   # fills at the next bar's open; 5% kept for costs
        elif not want_long and self.position:
            self.close()                             # also fills at the next bar's open


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()  # oldest first


def backtest(df: pd.DataFrame) -> pd.Series:
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(SLIPPAGE)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="daily", timeframe=bt.TimeFrame.Days)
    result = cerebro.run()[0]
    return pd.Series(result.analyzers.daily.get_analysis(), name="return").rename_axis("date")


def main() -> None:
    df = load_prices(DATA_CSV)
    print("HYPOTHETICAL simulated results on past data. Not a prediction or guarantee of future results.")
    # Each period is simulated on its own, so the test period starts flat and uses nothing from the train run.
    # The slow average needs SLOW bars to warm up in each period.
    for label, part in (("train", df.loc[:TRAIN_END]), ("test (look at this once, after the rule is final)",
                                                         df.loc[df.index > pd.Timestamp(TRAIN_END)])):
        net = backtest(part)
        equity = (1 + net).cumprod()
        print(f"{label}: total return {equity.iloc[-1] - 1:.2%}, "
              f"max drawdown {(equity / equity.cummax() - 1).min():.2%}")
    backtest(df).to_csv("returns.csv")


if __name__ == "__main__":
    main()

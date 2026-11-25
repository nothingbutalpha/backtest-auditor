"""Backtest Auditor template: pandas (one asset, daily bars, long or flat).

Timing conventions (keep them when you change the rule):
  1. The signal for bar t uses only data up to and including bar t's close.
  2. The trade fills at bar t+1's open: position = signal.shift(1).
  3. Costs are a parameter, charged on every change in position. Never set them to zero by default.
  4. Train and test are split by date. Choose parameters on the train period only, and count every
     variant you tried (rules, parameters, assets) as n_trials when you audit the results.

Input: a CSV you supply with columns date, open, high, low, close (split- and dividend-adjusted, all the
same way). Output: net returns per bar in returns.csv, ready for the auditor's audit_returns or
audit_backtest. Results are hypothetical, simulated past performance, not a prediction or guarantee.
"""
import pandas as pd

DATA_CSV = "prices.csv"      # your data: date, open, high, low, close
TRAIN_END = "2018-12-31"     # last date of the train period; everything after it is the test period
FAST, SLOW = 20, 100         # example parameters: replace with your own rule's
COST_BPS = 10.0              # all-in cost (commission, spread, slippage) per unit of one-way turnover, in bps


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()  # oldest first, so every rolling window looks back


def backtest(df: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    bt = df.copy()

    # --- Your rule goes here. Use data up to bar t's close only: trailing windows, no shift(-k),
    # no center=True, no backward fill, no statistics over all rows. Keep it inside this function,
    # where the auditor's check can follow the signal to the fill.
    # Example rule, to be replaced by yours: long when the fast moving average is above the slow one.
    fast = bt["close"].rolling(FAST).mean()
    slow = bt["close"].rolling(SLOW).mean()
    bt["signal"] = (fast > slow).astype(float)              # 1 = long, 0 = flat; known at bar t's close

    # --- Timing and costs: keep these lines as they are.
    bt["position"] = bt["signal"].shift(1).fillna(0.0)      # bought or sold at bar t+1's open
    bt["next_open_ret"] = bt["open"].shift(-1) / bt["open"] - 1  # outcome only: bar t's open to bar t+1's open
    bt["gross"] = bt["position"] * bt["next_open_ret"]      # a position filled at bar t's open earns this
    bt["turnover"] = bt["position"].diff().abs().fillna(bt["position"].abs())
    bt["cost"] = bt["turnover"] * cost_bps / 10_000         # charged on the bar of the fill
    bt["net"] = bt["gross"] - bt["cost"]
    return bt.iloc[:-1]                                      # the last bar has no next open yet


def summarize(net: pd.Series, label: str) -> None:
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    print(f"{label}: {net.index[0].date()} to {net.index[-1].date()}, {len(net)} bars, "
          f"total return {equity.iloc[-1] - 1:.2%}, max drawdown {drawdown.min():.2%}")


def main() -> None:
    bt = backtest(load_prices(DATA_CSV))
    train = bt.loc[:TRAIN_END]
    test = bt.loc[bt.index > pd.Timestamp(TRAIN_END)]
    print("HYPOTHETICAL simulated results on past data. Not a prediction or guarantee of future results.")
    summarize(train["net"], "train")
    summarize(test["net"], "test (look at this once, after the rule is final)")
    bt["net"].rename("return").rename_axis("date").to_csv("returns.csv")


if __name__ == "__main__":
    main()

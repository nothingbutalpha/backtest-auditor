# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
signal = (df["open"] < df["low"].shift(1)).astype(int)
intraday = df["close"] / df["open"] - 1
strat = signal * intraday - 0.0005 * signal
print("sharpe", strat.mean() / strat.std() * 252 ** 0.5)

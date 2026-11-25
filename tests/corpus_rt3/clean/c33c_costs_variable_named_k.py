# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
k = 0.0005
signal = (px > px.rolling(200).mean()).astype(int)
position = signal.shift(1).fillna(0)
strat = position * px.pct_change() - k * (position - position.shift(1)) ** 2
print("sharpe", strat.mean() / strat.std() * 252 ** 0.5)

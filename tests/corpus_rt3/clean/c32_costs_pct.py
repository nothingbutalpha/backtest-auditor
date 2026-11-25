# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
signal = (px > px.rolling(200).mean()).astype(int)
position = signal.shift(1).fillna(0)
gross = position * px.pct_change()
trade = position != position.shift(1)
net = gross - trade * 0.1 / 100
print("sharpe", net.mean() / net.std() * 252 ** 0.5)

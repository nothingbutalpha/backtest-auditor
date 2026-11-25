# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("spy.csv", parse_dates=["date"])
df["sma"] = df["close"].rolling(20).mean()
want_long = False
pnl = []
prev_close = None
holding = False
for row in df.itertuples():
    if holding and prev_close is not None:
        pnl.append(row.close / prev_close - 1)
    if want_long and not holding:
        holding = True
        pnl.append(row.close / row.open - 1 - 0.0005)
    want_long = row.close > row.sma
    if not want_long and holding:
        holding = False
    prev_close = row.close
print(sum(pnl))

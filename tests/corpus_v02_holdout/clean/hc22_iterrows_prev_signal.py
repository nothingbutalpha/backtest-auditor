# expect: none
# Looks like an iterrows same-bar trade, but each bar's return uses the position set on the prior bar.
import pandas as pd

df = pd.read_csv("ewz.csv", parse_dates=["date"])
df["ma20"] = df["close"].rolling(20).mean()

pos, prev_close, pnl, fee = 0, None, 0.0, 0.0008
for _, row in df.iterrows():
    if prev_close is not None:
        pnl += pos * (row["close"] / prev_close - 1)
    new_pos = 1 if row["close"] > row["ma20"] else 0
    if new_pos != pos:
        pnl -= fee
    pos, prev_close = new_pos, row["close"]

print(f"Cumulative net return: {pnl:.2%}")

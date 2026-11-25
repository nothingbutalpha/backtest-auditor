# expect: LA02
# Rows are sorted newest-first before the rolling mean, so each window averages later bars.
import pandas as pd

df = pd.read_csv("tsla.csv", parse_dates=["date"])
df = df.sort_values("date", ascending=False).reset_index(drop=True)
df["ma20"] = df["close"].rolling(20).mean()
df["signal"] = (df["close"] > df["ma20"]).astype(int)

df = df.sort_values("date").reset_index(drop=True)
df["ret"] = df["close"].pct_change()
df["pos"] = df["signal"].shift(1).fillna(0)
df["pnl"] = df["pos"] * df["ret"] - 0.0005 * df["pos"].diff().abs().fillna(0)
print("Total return:", (1 + df["pnl"]).prod() - 1)

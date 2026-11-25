# expect: none
# Looks like a reversed-sort window, but features are built ascending; descending sort is only for display.
import pandas as pd

df = pd.read_csv("tsla.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["ma20"] = df["close"].rolling(20).mean()
df["ret"] = df["close"].pct_change()
df["pos"] = (df["close"] > df["ma20"]).astype(int).shift(1).fillna(0)
df["pnl"] = df["pos"] * df["ret"] - 0.0005 * df["pos"].diff().abs().fillna(0)

recent = df.sort_values("date", ascending=False).head(10)
print(recent[["date", "close", "ma20", "pos", "pnl"]].to_string(index=False))
print("Total return:", (1 + df["pnl"].fillna(0)).prod() - 1)

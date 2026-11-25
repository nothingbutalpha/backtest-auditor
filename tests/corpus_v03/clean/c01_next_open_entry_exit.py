# expect: none
# Class 3: decide at the close of t, enter at the open of t+1, exit at the open of t+2.
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["sma50"] = df["close"].rolling(50).mean()
df["signal"] = (df["close"] > df["sma50"]).astype(int)
entry_px = df["open"].shift(-1)
exit_px = df["open"].shift(-2)
df["trade_ret"] = exit_px / entry_px - 1
cost = 0.0005
df["strategy"] = df["signal"] * df["trade_ret"] - cost * df["signal"].diff().abs()
print((1 + df["strategy"].fillna(0)).prod())

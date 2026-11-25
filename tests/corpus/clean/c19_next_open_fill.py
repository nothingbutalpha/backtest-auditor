# expect: none
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["sma"] = df["close"].rolling(50).mean()
df["signal"] = (df["close"] > df["sma"]).astype(int)
# decide at today's close, fill at tomorrow's open, exit at the following open
df["entry_price"] = df["open"].shift(-1)
df["exit_price"] = df["open"].shift(-2)
df["trade_ret"] = df["exit_price"] / df["entry_price"] - 1
df["strategy"] = df["signal"] * df["trade_ret"] - 0.0005 * df["signal"].diff().abs()

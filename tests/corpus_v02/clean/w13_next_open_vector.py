# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["sma"] = df["close"].rolling(20).mean()
df["signal"] = (df["close"] > df["sma"]).astype(int)
df["oo_ret"] = df["open"].shift(-1) / df["open"] - 1
df["strategy"] = df["signal"].shift(1) * df["oo_ret"] - 0.0005 * df["signal"].diff().abs()

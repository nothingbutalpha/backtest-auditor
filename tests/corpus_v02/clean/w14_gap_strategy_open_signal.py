# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["gap"] = df["open"] / df["close"].shift(1) - 1
df["signal"] = (df["gap"] < -0.01).astype(int)
df["intraday"] = df["close"] / df["open"] - 1
df["strategy"] = df["signal"] * df["intraday"] - 0.0005 * df["signal"]

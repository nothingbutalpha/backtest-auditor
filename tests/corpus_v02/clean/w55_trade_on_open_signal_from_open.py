# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["prev_close"] = df["close"].shift(1)
df["gap_down"] = (df["open"] < df["prev_close"] * 0.99).astype(int)
df["day_ret"] = (df["close"] - df["open"]) / df["open"]
df["strategy"] = df["gap_down"] * df["day_ret"] - 0.0005 * df["gap_down"]

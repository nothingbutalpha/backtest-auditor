import numpy as np
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["ret"] = df["close"].pct_change()
df["signal"] = (df["close"] > df["close"].rolling(10).mean()).astype(int)
df["strategy"] = df["signal"] * df["ret"] - 0.0005 * df["signal"].diff().abs()
print("Sharpe", np.sqrt(252) * df["strategy"].mean() / df["strategy"].std())

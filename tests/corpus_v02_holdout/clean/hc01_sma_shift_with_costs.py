# expect: none
# Looks like a same-bar SMA fill, but the position is lagged with shift(1) and costs are charged.
import numpy as np
import pandas as pd

df = pd.read_csv("xlk.csv", index_col="date", parse_dates=True)
df["sma"] = df["close"].rolling(50).mean()
df["signal"] = (df["close"] > df["sma"]).astype(int)
df["pos"] = df["signal"].shift(1).fillna(0)
df["ret"] = df["close"].pct_change().fillna(0)

cost = df["pos"].diff().abs().fillna(0) * (0.0005 + 0.0002)
df["net"] = df["pos"] * df["ret"] - cost
sharpe = df["net"].mean() / df["net"].std() * np.sqrt(252)
print(f"Net Sharpe: {sharpe:.2f}")

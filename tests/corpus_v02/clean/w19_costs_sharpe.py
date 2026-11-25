# expect: none
import numpy as np
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["fast"] = df["close"].rolling(10).mean()
df["slow"] = df["close"].rolling(50).mean()
df["position"] = (df["fast"] > df["slow"]).astype(int).shift(1)
df["gross"] = df["position"] * df["close"].pct_change()
df["net"] = df["gross"] - 0.0005 * df["position"].diff().abs()
sharpe = df["net"].mean() / df["net"].std() * np.sqrt(252)
print(f"Sharpe {sharpe:.2f}")

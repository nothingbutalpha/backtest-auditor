# expect: LA09
# RSI entry/exit thresholds are the 10th/90th percentiles of RSI over the entire sample.
import numpy as np
import pandas as pd

df = pd.read_csv("gld.csv", index_col="date", parse_dates=True)
delta = df["close"].diff()
up = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
down = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
df["rsi"] = 100 - 100 / (1 + up / down)

upper = df["rsi"].quantile(0.9)
lower = df["rsi"].quantile(0.1)
df["signal"] = np.select([df["rsi"] < lower, df["rsi"] > upper], [1, -1], default=0)
df["pos"] = df["signal"].shift(1).fillna(0)
df["net"] = df["pos"] * df["close"].pct_change() - 0.0005 * df["pos"].diff().abs()
print("Sharpe", df["net"].mean() / df["net"].std() * np.sqrt(252))

# expect: none
# Looks like a full-sample quantile threshold, but the RSI cutoffs come from the training window only.
import numpy as np
import pandas as pd

df = pd.read_csv("gld.csv", index_col="date", parse_dates=True)
delta = df["close"].diff()
up = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
down = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
df["rsi"] = 100 - 100 / (1 + up / down)

train = df.loc[:"2018-12-31"]
test = df.loc["2019-01-01":].copy()
upper, lower = train["rsi"].quantile(0.9), train["rsi"].quantile(0.1)

test["signal"] = np.select([test["rsi"] < lower, test["rsi"] > upper], [1, -1], default=0)
test["pos"] = test["signal"].shift(1).fillna(0)
test["net"] = test["pos"] * test["close"].pct_change() - 0.0005 * test["pos"].diff().abs()
print("Test Sharpe", test["net"].mean() / test["net"].std() * np.sqrt(252))

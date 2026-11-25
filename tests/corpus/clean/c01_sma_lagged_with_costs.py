# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["sma_fast"] = df["close"].rolling(20).mean()
df["sma_slow"] = df["close"].rolling(50).mean()
df["signal"] = np.where(df["sma_fast"] > df["sma_slow"], 1, 0)
df["position"] = df["signal"].shift(1)
df["ret"] = df["close"].pct_change()
cost_bps = 5
df["turnover"] = df["position"].diff().abs()
df["strategy"] = df["position"] * df["ret"] - df["turnover"] * cost_bps / 1e4
sharpe = df["strategy"].mean() / df["strategy"].std() * np.sqrt(252)

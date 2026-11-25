import numpy as np
import pandas as pd

COST_BPS = 10

df = pd.read_csv("features.csv", parse_dates=["date"], index_col="date")
position = df["signal"].shift(1).fillna(0)
turnover = position.diff().abs().fillna(0)
net = position * df["ret"] - turnover * COST_BPS / 1e4
sharpe = np.sqrt(252) * net.mean() / net.std()
print(f"net Sharpe {sharpe:.2f}")

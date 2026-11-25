# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["label"] = (df["close"].shift(-5) > df["close"]).astype(int)
df["target_ret"] = df["close"].pct_change(5).shift(-5)
rets = df["close"].pct_change().dropna()
sharpe = rets.mean() / rets.std() * np.sqrt(252)
drawdown = (1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1
equity = (1 + rets).cumprod()
equity_norm = equity / equity.iloc[0]

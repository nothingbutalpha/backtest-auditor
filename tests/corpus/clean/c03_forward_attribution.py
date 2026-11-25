# expect: none
# Signal at the close of t earns the return from t to t+1: multiplying by returns.shift(-1) is correct.
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["returns"] = df["close"].pct_change()
df["signal"] = np.sign(df["returns"].rolling(10).mean())
fee = 0.0005
df["strategy_ret"] = df["signal"] * df["returns"].shift(-1) - df["signal"].diff().abs() * fee

# expect: none
import numpy as np
import pandas as pd

px = pd.read_parquet("closes.parquet")
returns = px.pct_change()
vol = returns.rolling(63).std() * np.sqrt(252)
weights = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
held = weights.shift(1)
fee_bps = 2
portfolio = (held * returns).sum(axis=1) - held.diff().abs().sum(axis=1) * fee_bps / 1e4

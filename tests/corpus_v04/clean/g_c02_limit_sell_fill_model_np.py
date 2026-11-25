# expect: none
import numpy as np
import pandas as pd

df = pd.read_parquet("qqq.parquet")
delta = df["close"].diff()
rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
rsi = 100 - 100 / (1 + rs)
want_short = rsi > 70
limit_px = df["close"] * 1.01
filled = want_short & (df["high"].shift(-1) >= limit_px)
cover_px = df["close"].shift(-3)
pnl = np.where(filled, limit_px / cover_px - 1 - 0.001, 0.0)
print(pnl.sum(), filled.mean())

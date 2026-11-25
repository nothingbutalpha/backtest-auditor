# expect: none
import numpy as np
import pandas as pd

px = pd.read_parquet("silver.parquet")["close"]
x = px.to_numpy()
trail = np.convolve(x, np.ones(9) / 9, mode="valid")
trail = np.r_[np.full(8, np.nan), trail]
signal = pd.Series(np.where(x > trail, 1.0, -1.0), index=px.index)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())

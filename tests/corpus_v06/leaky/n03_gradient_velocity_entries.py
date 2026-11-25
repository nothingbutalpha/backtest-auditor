# expect: LA02:high@6
import numpy as np
import pandas as pd

close = pd.read_parquet("wheat_daily.parquet")["close"].to_numpy()
velocity = np.gradient(np.log(close))
entries = velocity > 0.002
pos = np.r_[0.0, entries[:-1].astype(float)]
ret = np.r_[0.0, np.diff(close) / close[:-1]]
print((pos * ret).sum())

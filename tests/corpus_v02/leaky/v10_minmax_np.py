# expect: LA09:medium
import numpy as np
import pandas as pd
df = pd.read_csv("gold.csv", index_col=0, parse_dates=True)
mom = df["close"].pct_change(30).to_numpy()
scaled = (mom - np.nanmin(mom)) / (np.nanmax(mom) - np.nanmin(mom))
df["signal"] = np.where(scaled > 0.7, 1, 0)
df["strat"] = df["signal"].shift(1) * df["close"].pct_change() - 0.0005 * df["signal"].diff().abs()

# leak: 5
import pandas as pd
closes = pd.read_parquet("crsp_wide.parquet")
last = closes.apply(pd.Series.last_valid_index)
keep = last[last == closes.index[-1]].index
universe = closes[keep]
mom = universe.pct_change(252).shift(21)
print(mom.rank(axis=1).iloc[-1])

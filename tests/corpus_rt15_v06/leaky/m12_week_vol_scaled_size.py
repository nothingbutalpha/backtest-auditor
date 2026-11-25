# leak: 5
import pandas as pd
h = pd.read_parquet("btc_1h.parquet")
h["ret"] = h["close"].pct_change()
h["wk_vol"] = h.groupby(pd.Grouper(freq="W"))["ret"].transform("std")
h["size"] = 0.01 / h["wk_vol"]
h["strategy_ret"] = h["size"].shift(1) * h["ret"]
print(h["strategy_ret"].sum())

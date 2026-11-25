# expect: none
import numpy as np
import pandas as pd

m = pd.read_parquet("cl_1min.parquet")
day = m.index.date
m["day_open"] = m.groupby(day)["open"].transform("first")
m["from_open"] = m["close"] / m["day_open"] - 1
m["signal"] = -np.sign(m["from_open"])
print((m["signal"].shift(1) * m["close"].pct_change()).sum())

# expect: LA12:medium@7
import numpy as np
import pandas as pd

m = pd.read_parquet("cl_1min.parquet")
day = m.index.date
m["day_last"] = m.groupby(day)["close"].transform("last")
m["gap_to_close"] = m["day_last"] / m["close"] - 1
m["signal"] = np.sign(m["gap_to_close"])
print((m["signal"].shift(1) * m["close"].pct_change()).sum())

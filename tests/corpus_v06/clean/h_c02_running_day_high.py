# expect: none
import pandas as pd

m = pd.read_parquet("cl_1min.parquet")
day = m.index.date
m["run_high"] = m.groupby(day)["high"].cummax()
m["breakout"] = (m["close"] >= m["run_high"].shift(1)).astype(float)
print((m["breakout"].shift(1) * m["close"].pct_change()).sum())

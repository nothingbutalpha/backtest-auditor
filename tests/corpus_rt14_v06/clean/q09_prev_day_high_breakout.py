# expect: none
import pandas as pd
m5 = pd.read_parquet("nq_5m.parquet")
day = m5.index.normalize()
daily_hi = m5.groupby(day)["high"].max().shift(1)
m5["prev_hi"] = daily_hi.reindex(day).to_numpy()
m5["signal"] = (m5["close"] > m5["prev_hi"]).astype(int)
m5["pnl"] = m5["signal"].shift(1) * m5["close"].pct_change()
print(m5["pnl"].sum())

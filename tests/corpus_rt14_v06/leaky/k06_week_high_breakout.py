# leak: 5
import pandas as pd
m5 = pd.read_parquet("nq_5m.parquet")
wk = m5.index.to_period("W")
m5["week_hi"] = m5.groupby(wk)["high"].transform("max")
m5["signal"] = (m5["close"] >= m5["week_hi"] * 0.995).astype(int)
m5["pnl"] = m5["signal"].shift(1) * m5["close"].pct_change()
print(m5["pnl"].sum())

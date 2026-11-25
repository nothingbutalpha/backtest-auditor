# expect: none
import pandas as pd
m1 = pd.read_parquet("es_1m.parquet")
m1["day_open"] = m1.groupby(m1.index.date)["open"].transform("first")
m1["run_hi"] = m1.groupby(m1.index.date)["high"].cummax()
m1["signal"] = ((m1["close"] > m1["day_open"]) & (m1["close"] >= m1["run_hi"])).astype(int)
m1["pnl"] = m1["signal"].shift(1) * m1["close"].pct_change()
print(m1["pnl"].sum())

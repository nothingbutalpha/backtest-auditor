# leak: 7
import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["ret"] = m["close"].pct_change()
m["signal"] = (m["close"] > m["close"].rolling(30).mean()).astype(int).shift(1)
m["pnl"] = m["signal"] * m["ret"]
m["day_total"] = m.groupby(m.index.date)["pnl"].transform("sum")
m["signal"] = m["signal"].where(m["day_total"] > -0.02, 0)
m["pnl"] = m["signal"] * m["ret"]
print(m["pnl"].sum())

# expect: none
import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["signal"] = (m["close"] > m["close"].rolling(20).mean()).astype(int).shift(1)
m["pnl"] = m["signal"] * m["close"].pct_change()
m["day_vol"] = m.groupby(m.index.date)["volume"].transform("sum")
m["vol_share"] = m["volume"] / m["day_vol"]
print(m["vol_share"].describe(), m["pnl"].sum())

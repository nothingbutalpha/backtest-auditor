# expect: none
import pandas as pd
m1 = pd.read_parquet("es_1m.parquet")
m1["signal"] = (m1["close"] > m1["close"].rolling(30).mean()).astype(int).shift(1)
m1["pnl"] = m1["signal"] * m1["close"].pct_change()
m1["day_pnl"] = m1.groupby(m1.index.date)["pnl"].transform("sum")
print(m1["day_pnl"].drop_duplicates().describe())

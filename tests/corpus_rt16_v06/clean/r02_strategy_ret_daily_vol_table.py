# expect: none
import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["position"] = (m["close"] > m["close"].rolling(60).mean()).astype(int).shift(1)
m["strategy_ret"] = m["position"] * m["close"].pct_change()
m["day_strategy_vol"] = m.groupby(m.index.date)["strategy_ret"].transform("std")
print(m["day_strategy_vol"].resample("D").last().describe())

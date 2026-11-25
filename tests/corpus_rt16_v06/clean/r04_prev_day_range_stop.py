# expect: none
import pandas as pd
m = pd.read_parquet("nq_1m.parquet")
day = m.index.normalize()
rng = (m.groupby(day)["high"].max() - m.groupby(day)["low"].min()).shift(1)
m["stop_dist"] = rng.reindex(day).to_numpy() * 0.5
m["signal"] = (m["close"] > m["open"]).astype(int).shift(1)
m["pnl"] = m["signal"] * m["close"].pct_change()
print(m["pnl"].sum(), m["stop_dist"].mean())

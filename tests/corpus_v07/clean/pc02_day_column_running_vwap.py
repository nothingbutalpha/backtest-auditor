# expect: none
import pandas as pd

m1 = pd.read_parquet("tsla_1m.parquet")
m1["session"] = m1.index.date
m1["notional"] = m1["close"] * m1["volume"]
run_vwap = m1.groupby("session")["notional"].cumsum() / m1.groupby("session")["volume"].cumsum()
m1["rich"] = m1["close"] > 1.003 * run_vwap
m1["fwd_20"] = m1.groupby("session")["close"].shift(-20) / m1["close"] - 1
print(m1.loc[m1["rich"], "fwd_20"].mean())

# expect: LA07:high@7
import pandas as pd

d = pd.read_parquet("amzn_daily_vwap.parquet")
d["long"] = (d["close"] > d["close"].rolling(30).mean()).astype(int)
d["vwap_ret"] = d["vwap"].shift(-1) / d["vwap"] - 1
d["strat"] = d["long"] * d["vwap_ret"]
print(d["strat"].mean() * 252)

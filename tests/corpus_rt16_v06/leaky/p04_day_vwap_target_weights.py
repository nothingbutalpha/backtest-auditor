# leak: 6
import pandas as pd
b = pd.read_parquet("spy_5m.parquet")
b["pv"] = b["close"] * b["volume"]
day = b.index.floor("D")
b["day_px"] = b.groupby(day)["close"].transform("mean")
b["weight"] = (b["day_px"] / b["close"] - 1).clip(-1, 1)
b["strat"] = b["weight"].shift(1) * b["close"].pct_change()
print(b["strat"].sum())

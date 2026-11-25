# leak: 6
import pandas as pd
ticks = pd.read_parquet("aapl_1m.parquet")
ticks["pv"] = ticks["close"] * ticks["volume"]
day = ticks.index.normalize()
ticks["day_mean"] = ticks.groupby(day)["close"].transform("mean")
ticks["signal"] = (ticks["close"] < ticks["day_mean"]).astype(int)
ticks["pnl"] = ticks["signal"].shift(1) * ticks["close"].pct_change()
print(ticks["pnl"].sum())

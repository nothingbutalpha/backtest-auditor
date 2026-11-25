# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
pv = (df["close"] * df["volume"]).rolling(20).sum()
vwap20 = pv / df["volume"].rolling(20).sum()
want_buy = (df["close"] < vwap20) & (df["close"].pct_change(5) < -0.03)
limit_px = df["close"] * 0.985
filled = want_buy & (df["low"].shift(-1) <= limit_px)
fill_px = np.minimum(df["open"].shift(-1), limit_px)
trade_ret = (df["close"].shift(-6) / fill_px - 1).where(filled) - 0.001
print(trade_ret.dropna().describe())

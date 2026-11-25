# expect: LA01|LA10:high
import pandas as pd

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
pv = (df["close"] * df["volume"]).rolling(20).sum()
vwap20 = pv / df["volume"].rolling(20).sum()
want_buy = (df["close"] < vwap20) & (df["low"].shift(-1) < df["close"] * 0.98)
limit_px = df["close"] * 0.99
filled = want_buy & (df["low"].shift(-1) <= limit_px)
trade_ret = (df["close"].shift(-5) / limit_px - 1)[filled] - 0.001
print(trade_ret.mean())

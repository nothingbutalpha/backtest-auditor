# expect: none
"""Buy-limit 1% under the close when the close is below the 20-day VWAP. The order rests during the next bar only:
it fills if the next bar's low reaches the limit. Exit at the close five bars later."""
import pandas as pd

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
pv = (df["close"] * df["volume"]).rolling(20).sum()
vwap20 = pv / df["volume"].rolling(20).sum()
want_buy = df["close"] < vwap20
limit_px = df["close"] * 0.99
filled = want_buy & (df["low"].shift(-1) <= limit_px)
exit_px = df["close"].shift(-5)
trade_ret = (exit_px / limit_px - 1)[filled] - 0.001
print(len(trade_ret), trade_ret.mean(), filled.sum() / want_buy.sum())

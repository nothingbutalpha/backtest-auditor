# expect: none
# A one-day forward return realigned by one day is today's return: lead 1, lag 1.
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["ret_today"] = df["fwd_ret"].shift(1)
df["sig"] = (df["ret_today"].rolling(5).mean() > 0).astype(int).shift(1)
df["pnl"] = df["sig"] * df["ret_today"] - 0.0005 * df["sig"].diff().abs()

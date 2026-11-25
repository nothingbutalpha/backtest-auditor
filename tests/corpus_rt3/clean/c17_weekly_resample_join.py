# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
daily = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
weekly = daily["close"].resample("W-FRI").last()
wk_ma = weekly.rolling(10).mean()
signal = (weekly > wk_ma).shift(1)
daily = daily.join(signal.rename("signal")).ffill()

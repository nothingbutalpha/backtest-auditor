# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
q_ret = pd.read_csv("quarterly_returns.csv", parse_dates=["date"])
f = pd.read_csv("fund.csv", parse_dates=["period_end"])
# contemporaneous accounting-return regression (Ball & Brown style), not a trading signal
m = q_ret.merge(f, left_on=["ticker", "date"], right_on=["ticker", "period_end"])
print(m[["ret", "eps_surprise"]].corr())

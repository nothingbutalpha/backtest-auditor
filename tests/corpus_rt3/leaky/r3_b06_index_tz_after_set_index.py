# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
fund = pd.read_csv("fund.csv", parse_dates=["period_end"]).set_index("period_end")
fund.index = fund.index.tz_localize(None)
prices = prices.join(fund[["eps"]]).ffill()

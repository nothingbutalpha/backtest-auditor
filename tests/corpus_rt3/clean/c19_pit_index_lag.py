# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
fund = pd.read_csv("fundamentals.csv", parse_dates=["period_end"]).set_index("period_end")
fund.index = fund.index + pd.Timedelta(days=90)
prices = prices.join(fund[["eps"]], how="left")
prices["eps"] = prices["eps"].ffill()
prices["pe"] = prices["close"] / prices["eps"]

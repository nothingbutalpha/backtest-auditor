# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
fund = pd.read_csv("fundamentals.csv", index_col="fiscalDateEnding", parse_dates=True)
fund = fund.shift(-1, freq="QE")
prices = prices.join(fund["eps"]).ffill()

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
fund = pd.read_csv("fundamentals.csv", index_col="fiscalDateEnding", parse_dates=True)
prices = prices.join(fund["eps"].shift(1, freq="QE")).ffill()

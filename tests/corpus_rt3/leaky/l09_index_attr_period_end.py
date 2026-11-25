# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
fund = pd.read_csv("fund.csv")
fund.index = pd.to_datetime(fund["fiscalDateEnding"])
prices["eps"] = fund["reportedEPS"].reindex(prices.index, method="ffill")
prices["signal"] = prices["eps"] / prices["close"]

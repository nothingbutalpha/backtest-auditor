# expect: none
import pandas as pd
fund = pd.read_csv("income_statements.csv", parse_dates=["period_end", "filing_date"])
guid = pd.read_csv("guidance.csv", parse_dates=["period_end"])
both = fund.merge(guid, on=["ticker", "period_end"], how="left")
both["surprise"] = both["eps"] - both["eps_guided"]
print(both.groupby("ticker")["surprise"].mean())

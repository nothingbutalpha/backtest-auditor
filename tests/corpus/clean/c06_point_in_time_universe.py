# expect: none
import pandas as pd

membership = pd.read_csv("sp500_membership_history.csv", parse_dates=["date"])
prices = pd.read_parquet("prices_incl_delisted.parquet")
filings = pd.read_csv("filings.csv", parse_dates=["accepted_at"]).sort_values("accepted_at")
prices = prices.sort_values("date")
panel = pd.merge_asof(prices, filings, left_on="date", right_on="accepted_at", by="ticker", direction="backward")
panel["book"] = panel.groupby("ticker")["book"].ffill()
panel = panel.merge(membership, on=["date", "ticker"], how="inner")

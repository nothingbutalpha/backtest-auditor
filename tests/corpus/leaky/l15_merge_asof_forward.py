# expect: LA04:high
import pandas as pd

prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
filings = pd.read_csv("filings.csv", parse_dates=["date"]).sort_values("date")
panel = pd.merge_asof(prices, filings, on="date", by="ticker", direction="forward")

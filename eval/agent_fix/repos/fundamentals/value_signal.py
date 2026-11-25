import pandas as pd

px = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date").sort_index()
filings = pd.read_csv("filings.csv", parse_dates=["filed"], index_col="filed")

daily = px.join(filings[["eps", "book"]], how="left").sort_index()
daily["eps"] = daily["eps"].bfill()
daily["book"] = daily["book"].fillna(method="backfill")
weekly = daily["close"].resample("W-FRI").last()
daily["week_close"] = weekly.reindex(daily.index, method="bfill")
daily["ey"] = daily["eps"] / daily["close"]
daily.to_csv("value.csv")

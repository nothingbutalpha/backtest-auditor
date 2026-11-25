import pandas as pd

px = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
filings = pd.read_csv("filings.csv", parse_dates=["filed"], index_col="filed")

daily = px.join(filings[["eps", "book"]], how="left")
daily[["eps", "book"]] = daily[["eps", "book"]].ffill()
daily["ey"] = daily["eps"] / daily["close"]
daily["ey"] = daily["ey"].fillna(0.0)
daily.to_csv("value.csv")

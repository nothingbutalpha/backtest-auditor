# expect: none
import pandas as pd
px = pd.read_csv("aapl.csv", index_col="date", parse_dates=True)
bs = pd.read_csv("aapl_balance.csv", parse_dates=["fiscalDateEnding"])
bs["available"] = bs["fiscalDateEnding"] + pd.Timedelta(days=90)
bs = bs.set_index("available").sort_index()
px["book"] = bs["totalShareholderEquity"].reindex(px.index, method="ffill")
px["bm"] = px["book"] / px["close"]

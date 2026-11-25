# expect: LA15:medium
import pandas as pd
px = pd.read_csv("aapl.csv", index_col="date", parse_dates=True)
bs = pd.read_csv("aapl_balance.csv", parse_dates=["fiscalDateEnding"])
bs = bs.set_index("fiscalDateEnding").sort_index()
px["book"] = bs["totalShareholderEquity"].reindex(px.index, method="ffill")
px["bm"] = px["book"] / px["close"]

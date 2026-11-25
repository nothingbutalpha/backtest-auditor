# expect: none
import pandas as pd
mem = pd.read_csv("index_membership.csv", parse_dates=["added", "removed"])
current = mem.query("removed.isnull()", engine="python")["ticker"].tolist()
px = pd.read_parquet("all_closes.parquet")[current]
last = px.iloc[-1]
mom = px.iloc[-1] / px.iloc[-253] - 1
orders = mom.nlargest(20).index
print("buy at tomorrow's open:", list(orders), last[orders].round(2).to_dict())

# expect: SB01@5
import pandas as pd

ref = pd.read_parquet("security_master.parquet")   # ticker, list_date, delist_date
alive = ref.query("delist_date.isnull()")["ticker"]
px = pd.read_parquet("us_close.parquet")
px = px[px.columns.intersection(alive)]
mom = px.pct_change(252).shift(21)
top = mom.rank(axis=1, ascending=False) <= 50
print(px.pct_change()[top.shift(1, fill_value=False)].mean(axis=1).mean())

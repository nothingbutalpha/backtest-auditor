# expect: none
import pandas as pd

px = pd.read_parquet("hk_close.parquet").sort_index()
tradable = px.columns[px.iloc[-1].notna()]
score = px[tradable].pct_change(60).iloc[-1]
orders = score.nlargest(10)
orders.to_csv("orders_today.csv")

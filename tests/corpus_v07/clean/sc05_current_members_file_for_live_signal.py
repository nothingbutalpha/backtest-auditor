# expect: none
import pandas as pd

syms = pd.read_csv("nasdaq100_members_latest.csv")["symbol"]
px = pd.read_parquet("us_close_recent.parquet")
px = px[px.columns.intersection(syms)]
signal = px.pct_change(126).iloc[-1].rank(ascending=False)
signal[signal <= 10].to_csv("buy_list.csv")

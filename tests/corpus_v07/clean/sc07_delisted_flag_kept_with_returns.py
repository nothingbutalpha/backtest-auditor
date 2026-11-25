# expect: none
import pandas as pd

ref = pd.read_parquet("security_master.parquet")
dead = ref.query("delist_date.notnull()")["ticker"]
px = pd.read_parquet("us_close.parquet")
print("delisted names in the panel:", px.columns.intersection(dead).size)

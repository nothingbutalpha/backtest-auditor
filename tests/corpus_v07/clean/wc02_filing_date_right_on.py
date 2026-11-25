# expect: none
import pandas as pd

px = pd.read_parquet("px_daily.parquet").sort_values("date")
fund = pd.read_parquet("fund_q.parquet").sort_values("filing_date")
df = pd.merge_asof(
    px,
    fund,
    left_on="date",
    right_on="filing_date",
    by="permno",
)
df["bm"] = df["book_equity"] / df["mktcap"]
print(df.groupby("date")["bm"].median().tail())

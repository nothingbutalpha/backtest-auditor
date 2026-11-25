# expect: LA15|LA04@10
import pandas as pd

px = pd.read_parquet("px_daily.parquet").sort_values("date")
fund = pd.read_parquet("fund_q.parquet").sort_values("fiscal_period_end")
df = pd.merge_asof(
    px,
    fund,
    left_on="date",
    right_on="fiscal_period_end",
    by="permno",
)
df["bm"] = df["book_equity"] / df["mktcap"]
df["bm_rank"] = df.groupby("date")["bm"].rank(pct=True)
print(df[df["bm_rank"] > 0.9].groupby("date").size().mean())

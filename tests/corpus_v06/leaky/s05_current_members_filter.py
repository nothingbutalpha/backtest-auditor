# expect: SB01:medium@5
import pandas as pd

hist = pd.read_csv("ftse100_membership.csv", parse_dates=["start", "end"])
live = hist[hist["end"].isna()]["ticker"].unique()
px = pd.read_parquet("uk_close.parquet")
px = px.loc["2005":, px.columns.intersection(live)]
m = px.resample("ME").last()
mom = m.shift(1) / m.shift(12) - 1
top = mom.rank(axis=1, ascending=False) <= 15
print(m.pct_change().where(top).mean(axis=1).mean() * 12)

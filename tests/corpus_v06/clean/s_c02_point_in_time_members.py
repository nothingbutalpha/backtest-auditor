# expect: none
import pandas as pd

hist = pd.read_csv("ftse100_membership.csv", parse_dates=["start", "end"])
px = pd.read_parquet("uk_close.parquet")
m = px.resample("ME").last()
rets = m.pct_change(fill_method=None)
out = {}
for d in m.index[12:]:
    live = hist[(hist["start"] <= d) & (hist["end"].isna() | (hist["end"] > d))]["ticker"]
    names = m.columns.intersection(live)
    mom = m.loc[:d, names].iloc[-2] / m.loc[:d, names].iloc[-13] - 1
    out[d] = mom.nlargest(15).index
print(len(out))

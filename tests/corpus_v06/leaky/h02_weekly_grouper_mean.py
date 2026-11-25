# expect: LA12:medium@5
import pandas as pd

d = pd.read_csv("nflx_daily.csv", index_col="date", parse_dates=True)
d["wk_avg_range"] = d.groupby(pd.Grouper(freq="W"))["range"].transform("mean")
d["quiet"] = d["range"] < 0.5 * d["wk_avg_range"]
d["pos"] = d["quiet"].astype(float).shift(1)
print((d["pos"] * d["close"].pct_change()).mean())

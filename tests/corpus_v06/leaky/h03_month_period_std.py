# expect: LA12:medium@6
import pandas as pd

d = pd.read_csv("tlt_daily.csv", index_col="date", parse_dates=True)
d["r"] = d["close"].pct_change()
d["m_vol"] = d.groupby(d.index.to_period("M"))["r"].transform("std")
d["weight"] = (0.006 / d["m_vol"]).clip(upper=3).shift(1)
print((d["weight"] * d["r"]).mean() * 252)

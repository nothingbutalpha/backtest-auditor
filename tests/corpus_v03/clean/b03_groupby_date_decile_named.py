# expect: none
# Class 2: groupby([date, decile])[fwd_ret].mean() assigned to neutral names, then a top-minus-bottom spread.
import numpy as np
import pandas as pd

panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
close = panel.groupby("ticker")["close"]
panel["value"] = panel["book"] / panel["mktcap"]
panel["fwd_ret"] = close.shift(-1) / panel["close"] - 1
panel["bucket"] = panel.groupby("date")["value"].rank(pct=True).mul(10).apply(np.ceil)
daily = panel.groupby(["date", "bucket"])["fwd_ret"].mean()
wide = daily.unstack("bucket")
top_minus_bottom = wide[10] - wide[1]
cost = 0.0010
net = top_minus_bottom - cost
print(net.mean() * 252, net.std() * np.sqrt(252))

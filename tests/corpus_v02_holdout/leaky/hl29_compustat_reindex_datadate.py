# expect: LA04
# Compustat quarterlies indexed by datadate are reindexed onto trading days with ffill.
import numpy as np
import pandas as pd

px = pd.read_csv("aapl.csv", index_col="date", parse_dates=True)
comp = pd.read_csv("compustat_q.csv", parse_dates=["datadate", "rdq"])
aapl = comp[comp["tic"] == "AAPL"].set_index("datadate").sort_index()

f = aapl[["niq", "ceqq", "cshoq"]].reindex(px.index, method="ffill")
px["roe"] = f["niq"] * 4 / f["ceqq"]
px["pb"] = px["close"] * f["cshoq"] / f["ceqq"]

px["pos"] = ((px["roe"] > 0.25) & (px["pb"] < 30)).astype(float).shift(1).fillna(0)
px["net"] = px["pos"] * px["close"].pct_change() - 0.0005 * px["pos"].diff().abs()
print("Sharpe", px["net"].mean() / px["net"].std() * np.sqrt(252))

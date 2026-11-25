# expect: LA12
# 5-minute bars use default left labels, then are reindexed and forward-filled onto 1-minute data.
import pandas as pd

m1 = pd.read_csv("nq_1min.csv", index_col="ts", parse_dates=True)
m5 = m1["close"].resample("5min").last().to_frame("close5")
m5["ema5"] = m5["close5"].ewm(span=12).mean()

m1 = m1.join(m5["ema5"].reindex(m1.index, method="ffill"))
m1["pos"] = (m1["close"] > m1["ema5"]).astype(int).shift(1).fillna(0)
m1["ret"] = m1["close"].pct_change().fillna(0)
m1["net"] = m1["pos"] * m1["ret"] - 0.00005 * m1["pos"].diff().abs().fillna(0)
print(m1["net"].sum())

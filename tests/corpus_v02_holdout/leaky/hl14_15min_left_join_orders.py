# expect: LA12
# 15-minute OHLC labelled left/closed left is forward-filled onto an order log by timestamp.
import pandas as pd

px = pd.read_csv("aapl_1min.csv", index_col="ts", parse_dates=True)
agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
bars = px.resample("15min", label="left", closed="left").agg(agg)
bars["range_pct"] = (bars["high"] - bars["low"]) / bars["open"]

orders = pd.read_csv("orders.csv", index_col="ts", parse_dates=True).sort_index()
orders = orders.join(bars["range_pct"].reindex(orders.index, method="ffill"))
orders["take"] = orders["range_pct"] < 0.004
fills = orders[orders["take"]]
print(len(fills), (fills["exit_px"] / fills["entry_px"] - 1 - 0.0004).mean())

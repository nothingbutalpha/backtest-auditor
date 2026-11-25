# expect: none
# Class 3: next-open prices under neutral names.
import pandas as pd

bars = pd.read_csv("qqq.csv", parse_dates=["date"], index_col="date")
nxt_open = bars["open"].shift(-1)
nxt2_open = bars["open"].shift(-2)
oo = nxt2_open / nxt_open - 1
pos = (bars["close"].pct_change(10) > 0).astype(int)
fee = 0.0003 * pos.diff().abs()
strategy = pos * oo - fee
print(strategy.sum())

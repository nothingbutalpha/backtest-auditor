# expect: LA10:high
import numpy as np
import pandas as pd

panel = pd.read_parquet("panel.parquet")
panel["fwd_ret"] = panel.groupby("ticker")["close"].pct_change().shift(-1)
mkt_next = panel.groupby("date")["fwd_ret"].mean()
bench = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"].pct_change().shift(-1)
strat = np.copysign(np.abs(bench), mkt_next - mkt_next.rolling(20).median())
print(strat.mean() * 252)

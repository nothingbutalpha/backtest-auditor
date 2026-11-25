# expect: LA10:high
import numpy as np
import pandas as pd

panel = pd.read_parquet("panel.parquet")
panel["fwd_ret"] = panel.groupby("ticker")["close"].pct_change().shift(-1)
mkt = panel.groupby("date")["fwd_ret"].mean()
tilt = np.sign(mkt.rolling(3).mean())
spy = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
frame = pd.DataFrame({"r": spy.pct_change().shift(-1)})
strat = frame.eval("r * @tilt")
print(strat.mean() * 252)

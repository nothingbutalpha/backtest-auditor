# leak: 6
import numpy as np
import pandas as pd
d = pd.read_csv("gold_daily.csv", index_col=0, parse_dates=True)
mon = pd.Grouper(freq="M")
d["fair"] = d.groupby(mon)["close"].transform("median")
d["exit_px"] = np.minimum(d["high"], d["fair"])
d["trade_ret"] = d["exit_px"] / d["open"] - 1
print(d["trade_ret"].mean())

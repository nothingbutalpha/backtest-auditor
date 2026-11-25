# expect: none
import numpy as np
import pandas as pd
d = pd.read_csv("strat.csv", index_col=0, parse_dates=True)
d["pnl"] = d["position"].shift(1) * d["ret"]
d["running_sharpe"] = d["pnl"].expanding(60).mean() / d["pnl"].expanding(60).std() * np.sqrt(252)
d["roll_dd"] = d["pnl"].cumsum() - d["pnl"].cumsum().rolling(126, min_periods=1).max()
print(d[["running_sharpe", "roll_dd"]].tail())

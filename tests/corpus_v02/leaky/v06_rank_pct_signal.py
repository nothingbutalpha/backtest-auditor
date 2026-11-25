# expect: LA09:medium
import numpy as np
import pandas as pd
df = pd.read_csv("btc.csv", index_col="date", parse_dates=True)
df["vol"] = df["close"].pct_change().rolling(20).std()
df["vol_rank"] = df["vol"].rank(pct=True)
df["signal"] = np.where(df["vol_rank"] < 0.3, 1, 0)
df["pnl"] = df["signal"].shift(1) * df["close"].pct_change() - 0.001 * df["signal"].diff().abs()

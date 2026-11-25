# expect: LA07
# Position is the sign of a trend that includes the current bar's return, times that same return.
import numpy as np
import pandas as pd

df = pd.read_csv("usdjpy_1h.csv", index_col="ts", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["trend"] = df["ret"].rolling(3).sum()
df["pos"] = np.sign(df["trend"]).fillna(0)
df["pnl"] = df["pos"] * df["ret"] - 0.00002 * df["pos"].diff().abs()

hit = (df["pnl"] > 0).mean()
print(f"hit rate {hit:.1%}, total {df['pnl'].sum():.3f}")

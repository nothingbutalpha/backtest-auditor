# expect: LA09:medium
import numpy as np
import pandas as pd

df = pd.read_csv("tlt.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(120)
df["rev"] = -df["close"].pct_change(10)
z = df[["mom", "rev"]].transform(lambda s: (s - s.mean()) / s.std())
df["signal"] = np.sign(z.sum(axis=1)).shift(1)
df["pnl"] = df["signal"] * df["close"].pct_change() - 0.0003 * df["signal"].diff().abs()
print(df["pnl"].mean() / df["pnl"].std() * 252 ** 0.5)

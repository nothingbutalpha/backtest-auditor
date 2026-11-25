# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("tlt.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(120)
df["z"] = df["mom"].rolling(252).apply(lambda w: (w[-1] - w.mean()) / w.std(), raw=True)
df["signal"] = np.sign(df["z"]).shift(1)
df["pnl"] = df["signal"] * df["close"].pct_change() - 0.0003 * df["signal"].diff().abs()
print(df["pnl"].mean())

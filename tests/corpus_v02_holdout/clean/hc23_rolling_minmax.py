# expect: none
# Looks like full-sample min-max scaling, but max/min come from a trailing 252-bar window.
import numpy as np
import pandas as pd

df = pd.read_csv("vix_spy.csv", index_col="date", parse_dates=True)
hi = df["vix"].rolling(252).max()
lo = df["vix"].rolling(252).min()
df["vix_n"] = (df["vix"] - lo) / (hi - lo)
df["vol_n"] = df["volume"] / df["volume"].rolling(252).max()

df["sig"] = ((df["vix_n"] > 0.8) & (df["vol_n"] > 0.5)).astype(float)
df["pos"] = df["sig"].shift(1).fillna(0)
df["net"] = df["pos"] * df["spy"].pct_change() - 0.0005 * df["pos"].diff().abs()
print("Sharpe", df["net"].mean() / df["net"].std() * np.sqrt(252))

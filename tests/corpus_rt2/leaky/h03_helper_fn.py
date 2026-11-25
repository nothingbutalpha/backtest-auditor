import pandas as pd
def make_feature(s):
    return s.shift(-5) / s - 1
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = make_feature(df["Close"])
df["signal"] = (df["mom"] > 0).astype(int)
df["strat"] = df["signal"].shift(1) * df["Close"].pct_change()

import pandas as pd

df = pd.read_csv("spy.csv", index_col=0, parse_dates=True).sort_index()
df["vix_signal"] = df["vix"].shift(-1)
df["position"] = (df["vix_signal"] < 20).astype(int)
df["strat"] = df["position"].shift(1) * df["close"].pct_change() - 0.0005

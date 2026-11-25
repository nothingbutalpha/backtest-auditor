import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
closes = df["Close"]
calls = {d: int(closes.iloc[k + 5] > closes.iloc[k]) for k, d in enumerate(closes.index[:-5])}
df["position"] = pd.Series(calls)
df["strat"] = df["position"].shift(1) * closes.pct_change()

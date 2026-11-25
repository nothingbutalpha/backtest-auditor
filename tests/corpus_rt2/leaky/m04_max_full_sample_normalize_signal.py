import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["Close"].pct_change(60)
df["score"] = df["mom"] / df["mom"].abs().max()
df["position"] = df["score"].clip(-1, 1)
df["pnl"] = df["position"].shift(1) * df["Close"].pct_change()

import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["vol"] = df["Close"].pct_change().rolling(20).std()
threshold = df["vol"].quantile(0.8)
df["position"] = (df["vol"] < threshold).astype(int)
df["strat"] = df["position"].shift(1) * df["Close"].pct_change()

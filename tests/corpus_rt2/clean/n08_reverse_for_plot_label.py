import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["fwd_max_20"] = df["Close"][::-1].rolling(20).max()[::-1]
df["mae_label"] = df["fwd_max_20"] / df["Close"] - 1
print(df["mae_label"].describe())

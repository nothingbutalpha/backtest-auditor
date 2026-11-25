import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
positions = []
for d in df.index:
    later = df.loc[d + pd.Timedelta(days=5):, "Close"]
    fut = later.iloc[0] if len(later) else df.loc[d, "Close"]
    positions.append(1 if fut > df.loc[d, "Close"] else 0)
df["pos"] = positions
df["pnl"] = df["pos"].shift(1) * df["Close"].pct_change()

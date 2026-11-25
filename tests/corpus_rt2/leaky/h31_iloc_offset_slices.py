import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
nxt = df["Close"].iloc[1:].reset_index(drop=True)
cur = df["Close"].iloc[:-1].reset_index(drop=True)
position = (nxt > cur).astype(int)
r = cur.pct_change()
strategy = position * r

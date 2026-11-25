import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
p = df["Close"].values
i, pos, pnl = 0, 0, 0.0
while i < len(p) - 1:
    nxt = p[i + 1]
    if nxt > p[i]:
        pos = 1
    else:
        pos = 0
    pnl += pos * (p[i + 1] - p[i])
    i += 1

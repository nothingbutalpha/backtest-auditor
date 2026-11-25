import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
c = df["Close"].tolist()
sig = [1 if b > a else -1 for a, b in zip(c[:-1], c[1:])]
rets = [b / a - 1 for a, b in zip(c[:-1], c[1:])]
pnl = [s * r for s, r in zip(sig, rets)]
print(sum(pnl))

# expect: none
import pandas as pd
closes = pd.read_csv("spy.csv")["close"].tolist()
fee = 0.0005
pnl = []
for prev, today, nxt in zip(closes[:-2], closes[1:-1], closes[2:]):
    s = 1 if today > prev else 0
    pnl.append(s * (nxt / today - 1) - fee * s)
print(sum(pnl))

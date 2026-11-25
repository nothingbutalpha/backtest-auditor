# expect: none
import pandas as pd
df = pd.read_csv("spy.csv")
closes = df["close"].to_numpy()
rets = df["close"].pct_change().fillna(0).to_numpy()
ma = df["close"].rolling(20).mean().to_numpy()
pos, equity, cost = 0, 1.0, 0.0005
for i in range(20, len(closes)):
    equity *= 1 + pos * rets[i]
    new_pos = 1 if closes[i] > ma[i] else 0
    if new_pos != pos:
        equity *= 1 - cost
    pos = new_pos
print(equity)

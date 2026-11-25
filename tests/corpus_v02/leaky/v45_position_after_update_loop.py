# expect: LA07:high
import pandas as pd
df = pd.read_csv("spy.csv")
closes = df["close"].to_numpy()
rets = df["close"].pct_change().fillna(0).to_numpy()
ma = df["close"].rolling(20).mean().to_numpy()
equity, cost = 1.0, 0.0005
for i in range(20, len(closes)):
    pos = 1 if closes[i] > ma[i] else 0
    equity *= 1 + pos * rets[i] - cost
print(equity)

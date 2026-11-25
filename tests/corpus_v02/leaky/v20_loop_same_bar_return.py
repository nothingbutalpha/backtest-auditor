# expect: LA07:high
import pandas as pd
df = pd.read_csv("spy.csv")
c = df["close"].to_numpy()
ret = df["close"].pct_change().fillna(0).to_numpy()
sma = df["close"].rolling(30).mean().to_numpy()
equity, fee = 1.0, 0.0005
for i in range(30, len(c)):
    if c[i] > sma[i]:
        equity *= 1 + ret[i] - fee
print(equity)

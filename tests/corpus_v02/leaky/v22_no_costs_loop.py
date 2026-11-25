# expect: CO01:possible
import pandas as pd
df = pd.read_csv("spy.csv")
closes = df["close"].tolist()
sma = df["close"].rolling(20).mean().tolist()
equity = [1.0]
for i in range(20, len(closes) - 1):
    held = closes[i] > sma[i]
    r = closes[i + 1] / closes[i] - 1 if held else 0.0
    equity.append(equity[-1] * (1 + r))
cagr = equity[-1] ** (252 / len(equity)) - 1
print(cagr)

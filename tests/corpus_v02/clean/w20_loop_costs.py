# expect: none
import pandas as pd
df = pd.read_csv("spy.csv")
closes = df["close"].tolist()
sma = df["close"].rolling(20).mean().tolist()
slippage = 0.0005
equity = [1.0]
for i in range(20, len(closes) - 1):
    held = closes[i] > sma[i]
    r = closes[i + 1] / closes[i] - 1 - slippage if held else 0.0
    equity.append(equity[-1] * (1 + r))
print(equity[-1])

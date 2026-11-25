# expect: none
import pandas as pd

df = pd.read_csv("prices.csv")
closes = df["close"].tolist()
sma = df["close"].rolling(50).mean().tolist()
cost = 0.0005
pnl = 0.0
for i in range(50, len(closes) - 1):
    if closes[i] > sma[i]:
        pnl += (closes[i + 1] - closes[i]) / closes[i] - cost

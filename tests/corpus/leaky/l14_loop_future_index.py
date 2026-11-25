# expect: LA13:high
import pandas as pd

df = pd.read_csv("prices.csv")
closes = df["close"].tolist()
cash, shares, fee = 10_000.0, 0, 1.0
for i in range(len(closes) - 1):
    if closes[i + 1] > closes[i] and shares == 0:
        shares = cash // closes[i]
        cash -= shares * closes[i] + fee
    elif shares > 0:
        cash += shares * closes[i] - fee
        shares = 0

import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
closes = df["Close"].tolist()
fee = 0.001
cash, shares = 10000.0, 0
for i in range(20, len(closes)):
    ma = sum(closes[i - 20:i]) / 20
    if closes[i - 1] > ma and shares == 0:
        shares = cash * (1 - fee) / closes[i]
        cash = 0
    elif closes[i - 1] < ma and shares > 0:
        cash = shares * closes[i] * (1 - fee)
        shares = 0

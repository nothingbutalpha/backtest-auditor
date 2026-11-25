import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
pos = 0
equity = [1.0]
for i in range(1, len(df)):
    r = df["Close"].iloc[i] / df["Close"].iloc[i - 1] - 1
    equity.append(equity[-1] * (1 + pos * r - 0.0005 * abs(pos)))
    pos = 1 if df["Close"].iloc[i] > df["Close"].iloc[max(0, i - 20):i + 1].mean() else 0

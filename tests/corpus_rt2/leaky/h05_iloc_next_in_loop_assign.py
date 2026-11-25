import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
cash, shares = 10000.0, 0
for i in range(len(df) - 1):
    tomorrow = df["Close"].iloc[i + 1]
    today = df["Close"].iloc[i]
    if tomorrow > today and shares == 0:
        shares = cash // today
        cash -= shares * today
    elif tomorrow < today and shares > 0:
        cash += shares * today
        shares = 0

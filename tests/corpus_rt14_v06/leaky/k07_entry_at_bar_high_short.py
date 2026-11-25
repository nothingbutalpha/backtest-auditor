# leak: 9
import pandas as pd
bars = pd.read_csv("cl_15m.csv")
trades = []
for i in range(1, len(bars) - 1):
    row = bars.iloc[i]
    prev = bars.iloc[i - 1]
    if row["close"] < prev["close"]:
        fill_price = row["high"]
        exit_price = bars.iloc[i + 1]["close"]
        trades.append(fill_price / exit_price - 1)
print(sum(trades))

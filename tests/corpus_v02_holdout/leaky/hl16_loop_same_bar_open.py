# expect: LA07
# Loop checks bar i's close against its SMA, then enters at bar i's open.
import pandas as pd

df = pd.read_csv("msft.csv", parse_dates=["date"])
close, opn = df["close"].values, df["open"].values
sma = df["close"].rolling(20).mean().values

trades = []
in_pos = False
entry = 0.0
for i in range(20, len(df)):
    if not in_pos and close[i] > sma[i]:
        entry = opn[i]
        in_pos = True
    elif in_pos and close[i] < sma[i]:
        trades.append(close[i] / entry - 1 - 0.001)
        in_pos = False

print("trades", len(trades), "avg", sum(trades) / max(len(trades), 1))

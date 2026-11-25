# expect: none
# Looks like the same-bar loop fill, but entries and exits happen at open[i + 1] with fees.
import pandas as pd

df = pd.read_csv("msft.csv", parse_dates=["date"])
close, opn = df["close"].values, df["open"].values
sma = df["close"].rolling(20).mean().values
fee = 0.001

trades = []
in_pos = False
entry = 0.0
for i in range(20, len(df) - 1):
    if not in_pos and close[i] > sma[i]:
        entry = opn[i + 1]
        in_pos = True
    elif in_pos and close[i] < sma[i]:
        trades.append(opn[i + 1] / entry - 1 - 2 * fee)
        in_pos = False

print("trades", len(trades), "avg", sum(trades) / max(len(trades), 1))

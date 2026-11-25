# expect: none
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_csv("spy.csv")
close = df["close"].to_numpy(); open_ = df["open"].to_numpy()
pos = 0; pnl = []
commission = 1.0
for i in range(20, len(close)):
    ma = close[i - 20:i].mean()
    if i + 1 < len(close) and close[i] > ma and pos == 0:
        pos = 1; entry = open_[i + 1]
    elif pos == 1 and close[i] < ma and i + 1 < len(close):
        pnl.append(open_[i + 1] - entry - commission); pos = 0

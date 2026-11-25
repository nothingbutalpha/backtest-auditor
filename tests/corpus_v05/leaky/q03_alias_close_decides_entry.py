# expect: LA13:high
import numpy as np
import pandas as pd

bars = pd.read_csv("spy_hourly.csv", index_col=0, parse_dates=True)
close = bars["close"]
atr = (bars["high"] - bars["low"]).rolling(14).mean()
results = []
in_pos, entry, target = False, None, None
for t in range(20, len(bars) - 1):
    nxt = bars.iloc[t + 1]
    if not in_pos and nxt["close"] > close.iloc[t]:
        entry = nxt["open"]
        target = entry + 1.5 * atr.iloc[t]
        in_pos = True
    elif in_pos:
        if nxt["high"] >= target:
            results.append(max(target, nxt["open"]) / entry - 1)
            in_pos = False
print(np.mean(results), len(results))

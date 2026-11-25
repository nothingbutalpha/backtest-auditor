# expect: LA07:high@10
import pandas as pd

bars = pd.read_csv("nvda_daily.csv", index_col="date", parse_dates=True)
bars["dip"] = bars["close"] < bars["close"].rolling(10).mean() * 0.96
out = []
for i in range(10, len(bars) - 3):
    bar = bars.iloc[i]
    if bar["dip"]:
        entry_px = bar["low"]
        out.append(bars["close"].iloc[i + 3] / entry_px - 1)
print(sum(out) / max(len(out), 1))

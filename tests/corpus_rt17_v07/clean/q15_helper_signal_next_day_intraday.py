# expect: none
import pandas as pd
def rule(d):
    return (d["close"] > d["close"].rolling(20).mean()).astype(float)
d = pd.read_csv("dia.csv", index_col=0, parse_dates=True)
pos = rule(d).shift(1)
pnl = pos * (d["close"] / d["open"] - 1)
print(pnl.sum())

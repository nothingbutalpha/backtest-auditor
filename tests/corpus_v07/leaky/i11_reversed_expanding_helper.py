# expect: LA09|LA02@5
import pandas as pd

def future_mean(x):
    return x[::-1].expanding(min_periods=30).mean()[::-1]

ratio = pd.read_csv("gold_silver.csv", index_col=0, parse_dates=True)["ratio"]
dev = ratio - future_mean(ratio)
pos = -dev.clip(-1, 1).shift(1)
print((pos * ratio.pct_change()).sum())

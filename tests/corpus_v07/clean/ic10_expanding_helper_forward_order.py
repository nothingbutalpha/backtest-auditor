# expect: none
import pandas as pd

def past_mean(x):
    return x.expanding(min_periods=30).mean()

ratio = pd.read_csv("gold_silver.csv", index_col=0, parse_dates=True)["ratio"]
dev = ratio - past_mean(ratio)
pos = -dev.clip(-1, 1).shift(1)
print((pos * ratio.pct_change()).sum())

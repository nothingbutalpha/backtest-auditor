# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
def zscore_oos(series, split):
    train = series.loc[:split]
    return (series - train.mean()) / train.std()
df = pd.read_csv("pair.csv", index_col=0, parse_dates=True)
spread = df["a"] - 1.2 * df["b"]
df["signal"] = -zscore_oos(spread, "2018").clip(-2, 2)
df["position"] = df["signal"].shift(1)

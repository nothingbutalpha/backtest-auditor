# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("pair.csv", index_col=0, parse_dates=True)
spread = df["a"] - 1.2 * df["b"]
train = spread.loc[:"2018"]
mu, sd = train.mean(), train.std()
signal = -((spread - mu) / sd).clip(-2, 2)
pos = signal.loc["2019":].shift(1)

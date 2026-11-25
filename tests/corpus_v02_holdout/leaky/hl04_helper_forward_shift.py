# expect: LA01
# Helper `forward` returns s.shift(-n); the entry signal is built from the next day's close.
import numpy as np
import pandas as pd


def forward(s, n=1):
    return s.shift(-n)


df = pd.read_csv("qqq.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["ema"] = df["close"].ewm(span=10).mean()
df["signal"] = np.where(forward(df["close"]) > df["ema"], 1, 0)
df["position"] = df["signal"].shift(1).fillna(0)
df["cost"] = df["position"].diff().abs().fillna(0) * 0.0005
df["strategy"] = df["position"] * df["ret"] - df["cost"]
print("Total return:", (1 + df["strategy"]).prod() - 1)

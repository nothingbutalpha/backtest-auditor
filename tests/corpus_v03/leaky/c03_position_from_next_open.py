# expect: LA01|LA10:high
# Twin of clean/c08: the position depends on the next open.
import numpy as np
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["next_open"] = df["open"].shift(-1)
df["ret_oo"] = df["open"].shift(-2) / df["next_open"] - 1
df["position"] = np.where(df["next_open"] > df["close"] * 1.01, 1, 0)
df["strategy"] = df["position"] * df["ret_oo"] - 0.0005

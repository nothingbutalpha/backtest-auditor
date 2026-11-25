# expect: none
# Class 3: open-to-open return stored as a column, multiplied by a close-based signal.
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["next_open"] = df["open"].shift(-1)
df["ret_oo"] = df["open"].shift(-2) / df["next_open"] - 1
df["signal"] = (df["close"].pct_change(5) < -0.02).astype(int)
df["strategy"] = df["signal"] * df["ret_oo"] - 0.0005 * df["signal"].diff().abs()
print(df["strategy"].sum())

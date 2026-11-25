# expect: LA01|LA10:high
# Twin of clean/e03: a 5-day forward return lagged by only 1 day still sees 4 future days.
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["fwd5"] = df["close"].pct_change(5).shift(-5)
df["sig"] = (df["fwd5"].shift(1) > 0).astype(int)
df["pnl"] = df["sig"] * df["close"].pct_change() - 0.0005 * df["sig"].diff().abs()

# expect: LA09:medium
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_csv("spy.csv")
df["ret"] = df["close"].pct_change()
df["signal"] = (df["ret"].rolling(5).mean() > df["ret"].mean()).astype(int)
df["strategy"] = df["signal"].shift(1) * df["ret"] - 0.0005 * df["signal"].diff().abs()

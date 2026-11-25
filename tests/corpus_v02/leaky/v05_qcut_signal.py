# expect: LA09:medium
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(60)
df["quintile"] = pd.qcut(df["mom"], 5, labels=False)
df["position"] = (df["quintile"] == 4).astype(int)
df["strategy"] = df["position"].shift(1) * df["close"].pct_change() - 0.0005 * df["position"].diff().abs()

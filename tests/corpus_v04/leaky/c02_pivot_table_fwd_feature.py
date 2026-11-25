# expect: LA10|LA01:high
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret_fwd_5d"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["mom"] = df.groupby("ticker")["close"].pct_change(60)
tbl = df.pivot_table(index="date", columns="ticker", values="ret_fwd_5d", aggfunc="mean")
features = tbl.rolling(3).mean().stack().rename("f").to_frame()
features["mom"] = df.set_index(["date", "ticker"])["mom"]
label = df.set_index(["date", "ticker"])["ret_fwd_5d"].reindex(features.index)
ok = features.notna().all(axis=1) & label.notna()
model = Ridge().fit(features[ok].iloc[:50000], label[ok].iloc[:50000])
print(model.score(features[ok].iloc[50005:], label[ok].iloc[50005:]))

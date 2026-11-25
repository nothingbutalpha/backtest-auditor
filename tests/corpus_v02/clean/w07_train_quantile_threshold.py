# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(20)
train = df.loc[:"2015"]
thr = train["mom"].quantile(0.8)
test = df.loc["2016":].copy()
test["signal"] = (test["mom"] > thr).astype(int)
test["strategy"] = test["signal"].shift(1) * test["close"].pct_change() - 0.0005 * test["signal"].diff().abs()

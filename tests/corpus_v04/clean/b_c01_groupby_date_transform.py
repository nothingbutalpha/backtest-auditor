# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

panel = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(120)
panel["rev"] = -panel.groupby("ticker")["close"].pct_change(5)
panel["fwd_ret"] = panel.groupby("ticker")["close"].shift(-5) / panel["close"] - 1
cols = ["mom", "rev"]
panel[cols] = panel.groupby("date")[cols].transform(lambda s: (s - s.mean()) / s.std())
train = panel[panel["date"] < "2018-01-01"].dropna()
test = panel[panel["date"] >= "2018-01-06"].dropna()
model = Ridge().fit(train[cols], train["fwd_ret"])
test["score"] = model.predict(test[cols])
print(test.groupby("date")[["score", "fwd_ret"]].corr(method="spearman").iloc[0::2, -1].mean())

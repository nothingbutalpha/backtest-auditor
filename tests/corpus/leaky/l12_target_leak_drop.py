# expect: LA10:high
# The feature matrix drops `target` but keeps `fwd_ret`, from which the target was built.
import pandas as pd
from lightgbm import LGBMClassifier

df = pd.read_csv("panel.csv", parse_dates=["date"])
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["target"] = (df["fwd_ret"] > 0).astype(int)
df["mom"] = df.groupby("ticker")["close"].pct_change(20)
X = df.drop(columns=["target", "date", "ticker"])
y = df["target"]
cut = df["date"] < "2020-01-01"
model = LGBMClassifier().fit(X[cut], y[cut])

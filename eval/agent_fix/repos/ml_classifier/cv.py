import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("daily.csv", parse_dates=["date"]).sort_values("date")
df["ret_1"] = df["close"].pct_change()
df["ret_20"] = df["close"].pct_change(20)
df["target"] = df["close"].pct_change().shift(-1)
df = df.dropna()

cv = KFold(n_splits=5, shuffle=True, random_state=7)
scores = cross_val_score(Ridge(alpha=1.0), df[["ret_1", "ret_20"]], df["target"], cv=cv)
print("mean CV R^2", scores.mean())

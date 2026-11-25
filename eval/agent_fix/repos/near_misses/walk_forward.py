import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("daily.csv", parse_dates=["date"]).sort_values("date")
df["ret_1"] = df["close"].pct_change()
df["target"] = df["close"].pct_change().shift(-1)
df = df.dropna()
X, y = df[["ret_1"]], df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler().fit(X_train)
model = Ridge().fit(scaler.transform(X_train), y_train)
print("test R^2", model.score(scaler.transform(X_test), y_test))
print("walk-forward", cross_val_score(Ridge(), X, y, cv=TimeSeriesSplit(n_splits=5)).mean())

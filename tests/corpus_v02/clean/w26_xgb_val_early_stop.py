# expect: none
import pandas as pd
from xgboost import XGBClassifier
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
n = len(df)
X, y = df.drop(columns="target"), df["target"]
X_train, y_train = X.iloc[: int(n * 0.6)], y.iloc[: int(n * 0.6)]
X_val, y_val = X.iloc[int(n * 0.6): int(n * 0.8)], y.iloc[int(n * 0.6): int(n * 0.8)]
X_test, y_test = X.iloc[int(n * 0.8):], y.iloc[int(n * 0.8):]
model = XGBClassifier(n_estimators=2000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("test accuracy", (model.predict(X_test) == y_test).mean())

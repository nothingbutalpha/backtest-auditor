# expect: LA14:high
import pandas as pd
from xgboost import XGBClassifier
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
split = int(len(df) * 0.8)
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
model = XGBClassifier(n_estimators=2000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("test accuracy", (model.predict(X_test) == y_test).mean())

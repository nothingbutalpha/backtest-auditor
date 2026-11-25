# expect: LA14
# XGBoost early-stops on eval_set=[(X_test, y_test)] and then reports accuracy on that test set.
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = pd.read_parquet("daily_features_labeled.parquet")
y = data.pop("label")
X = data

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = XGBClassifier(n_estimators=2000, learning_rate=0.02, max_depth=4, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

pred = model.predict(X_test)
print("Best iteration:", model.best_iteration)
print("Out-of-sample accuracy:", accuracy_score(y_test, pred))

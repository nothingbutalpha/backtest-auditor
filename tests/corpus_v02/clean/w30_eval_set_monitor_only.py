# expect: none
import pandas as pd
from xgboost import XGBRegressor
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
cut = int(len(df) * 0.8)
X_train, X_test = df.iloc[:cut, :-1], df.iloc[cut:, :-1]
y_train, y_test = df.iloc[:cut, -1], df.iloc[cut:, -1]
model = XGBRegressor(n_estimators=300, max_depth=3)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
print(model.score(X_test, y_test))

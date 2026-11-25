# expect: LA05|LA06|LA14@9
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_parquet("power_features.parquet").sort_index()
X, y = df.drop(columns="price_fwd"), df["price_fwd"]
gs = GridSearchCV(GradientBoostingRegressor(), {"max_depth": [2, 3, 5]}, cv=TimeSeriesSplit(4))
gs.fit(X, y)
start = int(len(X) * 0.8)
oos = gs.best_estimator_.predict(X.iloc[start:])
print(pd.Series(oos, index=X.index[start:]).corr(y.iloc[start:]))

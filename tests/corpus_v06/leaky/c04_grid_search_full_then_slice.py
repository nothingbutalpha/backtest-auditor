# expect: LA14|LA05:medium@9
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

df = pd.read_parquet("macro_features.parquet").sort_index()
X, y = df.drop(columns="fwd_ret_1m"), df["fwd_ret_1m"]
gs = GridSearchCV(Lasso(), {"alpha": [1e-4, 1e-3, 1e-2]}, cv=TimeSeriesSplit(5))
gs.fit(X, y)
X_test, y_test = X.loc["2021":], y.loc["2021":]
print("test R2", gs.best_estimator_.score(X_test, y_test))

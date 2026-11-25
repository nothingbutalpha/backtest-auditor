# expect: none
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

df = pd.read_parquet("macro_features.parquet").sort_index()
X, y = df.drop(columns="fwd_ret_1m"), df["fwd_ret_1m"]
X_train, y_train = X.loc[:"2020-11"], y.loc[:"2020-11"]
gs = GridSearchCV(Lasso(), {"alpha": [1e-4, 1e-3, 1e-2]}, cv=TimeSeriesSplit(5))
gs.fit(X_train, y_train)
print("test R2", gs.best_estimator_.score(X.loc["2021":], y.loc["2021":]))

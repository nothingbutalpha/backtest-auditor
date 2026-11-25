# expect: LA06:possible
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("features.csv", parse_dates=["date"], index_col="date")
X_train, y_train = df.loc[:"2018", ["mom", "vol"]], df.loc[:"2018", "fwd_ret"]
search = GridSearchCV(RandomForestRegressor(), {"max_depth": [2, 4, 8]}, cv=5)
search.fit(X_train, y_train)

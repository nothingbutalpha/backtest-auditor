# leak: 9
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
tbl = pd.read_parquet("macro_panel.parquet")
y = tbl["spx_ret_1m"].shift(-1).dropna()
X = tbl.loc[y.index].drop(columns="spx_ret_1m")
lasso = LassoCV(cv=5).fit(X, y)
chosen = X.columns[np.abs(lasso.coef_) > 1e-6]
preds = []
for k in range(120, len(X)):
    lr = LinearRegression().fit(X[chosen].iloc[:k], y.iloc[:k])
    preds.append(lr.predict(X[chosen].iloc[[k]])[0])
print(np.corrcoef(preds, y.iloc[120:])[0, 1])

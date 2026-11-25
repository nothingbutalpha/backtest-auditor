# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
from sklearn.linear_model import Ridge
res = {}
for d in pd.date_range("2015", "2023", freq="YS"):
    X_train, y_train = X.loc[: d - pd.Timedelta(days=1)], y.loc[: d - pd.Timedelta(days=1)]
    X_test, y_test = X.loc[d: d + pd.DateOffset(years=1)], y.loc[d: d + pd.DateOffset(years=1)]
    res[d.year] = Ridge().fit(X_train, y_train).score(X_test, y_test)
s = pd.Series(res)
print(s.describe(), "best year", s.idxmax())

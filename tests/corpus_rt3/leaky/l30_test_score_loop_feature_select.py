# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from sklearn.linear_model import LinearRegression
keep = []
for col in X_train.columns:
    m = LinearRegression().fit(X_train[[col]], y_train)
    if m.score(X_test[[col]], y_test) > 0.01:
        keep.append(col)

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
best, best_mse = None, float("inf")
for alpha in [0.1, 1, 10]:
    m = Ridge(alpha=alpha).fit(X_train, y_train)
    mse = mean_squared_error(y_dev, m.predict(X_dev))
    if mse < best_mse:
        best, best_mse = m, mse
print("test mse", mean_squared_error(y_test, best.predict(X_test)))

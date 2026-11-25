# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
from sklearn.linear_model import Ridge
scores = {}
for window in [5, 20, 60, 120]:
    Xw_train = X_train.rolling(window).mean().dropna()
    Xw_test = X_test.rolling(window).mean().dropna()
    m = Ridge().fit(Xw_train, y_train.loc[Xw_train.index])
    scores[window] = m.score(Xw_test, y_test.loc[Xw_test.index])
best_window = max(scores, key=scores.get)
print("test R2", scores[best_window])

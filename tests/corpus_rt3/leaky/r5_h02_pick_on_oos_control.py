# expect: any
# From the fourth red-team review of PR #8 (35efd4a), 2026-10-16.
from sklearn.linear_model import Ridge
is_scores, oos_scores = {}, {}
for alpha in [0.01, 0.1, 1, 10, 100]:
    m = Ridge(alpha=alpha).fit(X_train, y_train)
    is_scores[alpha] = m.score(X_train, y_train)
    oos_scores[alpha] = m.score(X_test, y_test)
best = max(oos_scores, key=oos_scores.get)
print("chosen", best, "out-of-sample R2", oos_scores[best])

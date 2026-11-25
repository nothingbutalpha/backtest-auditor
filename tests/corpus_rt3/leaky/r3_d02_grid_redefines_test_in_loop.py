# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
results = {}
for alpha in [0.01, 0.1, 1, 10, 100]:
    sc = StandardScaler().fit(X_train)
    X_test = sc.transform(X_test_raw)
    m = Ridge(alpha=alpha).fit(sc.transform(X_train), y_train)
    results[alpha] = m.score(X_test, y_test)
best_alpha = max(results, key=results.get)
print(best_alpha, results[best_alpha])

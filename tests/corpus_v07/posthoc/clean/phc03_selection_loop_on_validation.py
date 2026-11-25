# expect: none
import numpy as np
from sklearn.ensemble import RandomForestRegressor

X = np.load("X.npy"); y = np.load("y.npy")
n = len(X)
a, b = int(n * 0.6), int(n * 0.8)
X_tr, X_val, X_test = X[:a], X[a:b], X[b:]
y_tr, y_val, y_test = y[:a], y[a:b], y[b:]

best, best_score = None, -np.inf
for depth in [2, 4, 6, 8]:
    m = RandomForestRegressor(max_depth=depth, random_state=0)
    m.fit(X_tr, y_tr)

    s = m.score(X_val, y_val)
    if s > best_score:
        best, best_score = m, s
print("validation R2:", best_score, "test R2 (once):", best.score(X_test, y_test))

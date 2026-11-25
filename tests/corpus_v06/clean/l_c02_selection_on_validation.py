# expect: none
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.load("X.npy"); y = np.load("y.npy")
a, b = int(len(X) * 0.6), int(len(X) * 0.8)
X_tr, X_val, X_te = X[:a], X[a:b], X[b:]
y_tr, y_val, y_te = y[:a], y[a:b], y[b:]
best_k, best = None, -np.inf
for k in [5, 10, 20, 50, 100]:
    model = KNeighborsRegressor(n_neighbors=k).fit(X_tr, y_tr)
    r2 = model.score(X_val, y_val)
    if r2 > best:
        best_k, best = k, r2
final = KNeighborsRegressor(n_neighbors=best_k).fit(X[:b], y[:b])
print("test R2", final.score(X_te, y_te))

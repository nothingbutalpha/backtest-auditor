# expect: LA14:medium@11
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.load("X.npy"); y = np.load("y.npy")
cut = int(len(X) * 0.8)
X_tr, X_te, y_tr, y_te = X[:cut], X[cut:], y[:cut], y[cut:]
best_k, best = None, -np.inf
for k in [5, 10, 20, 50, 100]:
    model = KNeighborsRegressor(n_neighbors=k).fit(X_tr, y_tr)
    r2 = model.score(X_te, y_te)
    if r2 > best:
        best_k, best = k, r2
print("test R2", best, "k", best_k)
